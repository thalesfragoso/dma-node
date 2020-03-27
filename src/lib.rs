#![no_std]

use as_slice::{AsMutSlice, AsSlice};
use core::{
    default::Default,
    fmt,
    mem::MaybeUninit,
    ops::{Deref, DerefMut, Drop},
    ptr, slice,
};
use generic_array::{typenum::marker_traits::Unsigned, ArrayLength, GenericArray};

pub mod typenum {
    pub use generic_array::typenum::consts;
}

pub trait DMANode<T>: Deref<Target = [T]> + DerefMut {
    /// Creates a new node
    fn new() -> Self;

    /// Gives a `&mut [W]` slice to write into with the maximum size, the `commit` method
    /// must then be used to set the actual number of bytes written.
    ///
    /// Note that this function internally first initializes to default the non-initialized elements
    /// of the node's buffer.
    fn write(&mut self) -> &mut [T];

    /// Used to shrink the current size of the slice in the node, mostly used in conjunction
    /// with `write`.
    fn commit(&mut self, shrink_to: usize);

    /// Used to write data into the node, and returns how many bytes were written from `buf`.
    ///
    /// If the node is already partially filled, this will continue filling the node.
    fn write_slice(&mut self, buf: &[T]) -> usize;

    /// Clear the node of all data making it empty.
    fn clear(&mut self);

    /// Reads how many bytes are available.
    fn len(&self) -> usize;

    /// Checks if the node is empty.
    fn is_empty(&self) -> bool;

    /// Sets the length of the internal buffer.
    ///
    /// # Safety
    ///
    /// The user has to ensure that the length is valid and all elements in that length have been
    /// initialized.
    unsafe fn set_len(&mut self, len: usize);

    /// Returns the address of the buffer.
    fn buffer_address_for_dma(&self) -> usize;

    /// Returns the maximum length of the internal buffer.
    fn max_len(&self) -> usize;

    /// Returns the number of free elements in the internal buffer
    #[inline]
    fn free(&self) -> usize {
        self.max_len() - self.len()
    }
}

pub struct Node<N, W>
where
    N: ArrayLength<MaybeUninit<W>> + Unsigned + 'static,
{
    len: usize,
    buf: GenericArray<MaybeUninit<W>, N>,
}

// Heavily inspired by korken89 work
impl<N, W> DMANode<W> for Node<N, W>
where
    N: ArrayLength<MaybeUninit<W>> + Unsigned + 'static,
    W: Default,
{
    fn new() -> Self {
        Self {
            len: 0,
            buf: unsafe {
                #[allow(clippy::uninit_assumed_init)]
                MaybeUninit::uninit().assume_init()
            },
        }
    }

    fn write(&mut self) -> &mut [W] {
        // Initialize memory with a safe value
        for elem in self.buf.iter_mut().skip(self.len) {
            unsafe {
                ptr::write(elem.as_mut_ptr() as *mut W, W::default());
            }
        }
        self.len = N::USIZE; // Set to max so `commit` may shrink it if needed

        unsafe {
            slice::from_raw_parts_mut(self.buf.as_mut_slice().as_mut_ptr() as *mut _, N::USIZE)
        }
    }

    fn commit(&mut self, shrink_to: usize) {
        // Only shrinking is allowed to remain safe with the `MaybeUninit`
        if shrink_to < self.len {
            self.len = shrink_to;
        }
    }

    fn write_slice(&mut self, buf: &[W]) -> usize {
        let count = buf.len().min(self.free());

        // Used to write data into the `MaybeUninit`, safe based on the size check above
        unsafe {
            ptr::copy_nonoverlapping(
                buf.as_ptr(),
                self.buf.as_mut_slice().as_mut_ptr().add(self.len) as *mut W,
                count,
            );
        }

        self.len += count;
        count
    }

    #[inline]
    fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    unsafe fn set_len(&mut self, len: usize) {
        self.len = len;
    }

    #[inline]
    fn buffer_address_for_dma(&self) -> usize {
        self.buf.as_slice().as_ptr() as usize
    }

    #[inline]
    fn max_len(&self) -> usize {
        N::USIZE
    }
}

impl<N, W> Node<N, W>
where
    N: ArrayLength<MaybeUninit<W>> + Unsigned + 'static,
{
    /// Gives the underling buffer to be modified and the already initialized length, the user is
    /// is free to modify it, but must return the correct number of uninitialized elements that
    /// were initialized.
    ///
    /// # Safety
    ///
    /// The user must provide the correct number of newer initialized elements, otherwise there will
    /// be a risk of accessing uninitialized data, which is undefined behavior.
    pub unsafe fn write_with(
        &mut self,
        f: impl FnOnce(&mut GenericArray<MaybeUninit<W>, N>, usize) -> usize,
    ) {
        let count = f(&mut self.buf, self.len);
        self.len = if count + self.len > N::USIZE {
            N::USIZE
        } else {
            self.len + count
        }
    }
}

impl<N, W> Deref for Node<N, W>
where
    N: ArrayLength<MaybeUninit<W>> + Unsigned + 'static,
{
    type Target = [W];

    fn deref(&self) -> &Self::Target {
        // Safe as it uses the internal length of valid data
        unsafe {
            slice::from_raw_parts(self.buf.as_slice().as_ptr() as *const _, self.len as usize)
        }
    }
}

impl<N, W> DerefMut for Node<N, W>
where
    N: ArrayLength<MaybeUninit<W>> + Unsigned + 'static,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Safe as it uses the internal length of valid data
        unsafe {
            slice::from_raw_parts_mut(
                self.buf.as_mut_slice().as_ptr() as *mut _,
                self.len as usize,
            )
        }
    }
}

impl<N, W> Drop for Node<N, W>
where
    N: ArrayLength<MaybeUninit<W>> + Unsigned + 'static,
{
    fn drop(&mut self) {
        for elem in self.iter_mut() {
            unsafe {
                ptr::drop_in_place(elem);
            }
        }
    }
}

impl<N> fmt::Write for Node<N, u8>
where
    N: ArrayLength<MaybeUninit<u8>> + Unsigned + 'static,
{
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let free = self.free();

        if s.len() > free {
            Err(fmt::Error)
        } else {
            self.write_slice(s.as_bytes());
            Ok(())
        }
    }
}

impl<N, W> fmt::Debug for Node<N, W>
where
    N: ArrayLength<MaybeUninit<W>> + Unsigned + 'static,
    W: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", &self[..])
    }
}

impl<N, W> AsSlice for Node<N, W>
where
    N: ArrayLength<MaybeUninit<W>> + Unsigned + 'static,
{
    type Element = W;

    fn as_slice(&self) -> &[Self::Element] {
        &self[..]
    }
}

impl<N, W> AsMutSlice for Node<N, W>
where
    N: ArrayLength<MaybeUninit<W>> + Unsigned + 'static,
{
    fn as_mut_slice(&mut self) -> &mut [Self::Element] {
        &mut self[..]
    }
}

#[cfg(test)]
mod tests {

    use crate::typenum::consts::*;
    use crate::{DMANode, Node};
    use core::{fmt::Write, ptr};

    const DATA: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8];

    #[test]
    fn write_read() {
        let mut node = Node::<U8, u8>::new();
        let written = node.write_slice(DATA);
        assert_eq!(written, DATA.len());
        assert_eq!(node.len(), DATA.len());
        assert_eq!(&node[..], DATA);
        assert_eq!(node.max_len(), 8);
    }

    #[test]
    fn write_commit() {
        let mut node = Node::<U9, u8>::new();
        let inner = node.write();
        for (elem, data) in inner.iter_mut().zip(DATA.iter()) {
            *elem = *data;
        }
        node.commit(DATA.len());
        assert_eq!(&node[..], DATA);
    }

    #[test]
    fn write_with() {
        let mut node = Node::<U8, u8>::new();
        let max_len = core::cmp::min(node.max_len(), DATA.len());
        unsafe {
            node.write_with(|buf, _len| {
                ptr::copy_nonoverlapping(DATA.as_ptr(), buf.as_mut_ptr() as *mut u8, max_len);
                max_len
            });
        }
        assert_eq!(&node[..], DATA);
    }

    #[test]
    fn fmt_write() {
        let text = "olá";
        let text2 = "oi";
        let mut node = Node::<U8, u8>::new();
        write!(node, "{}", text).unwrap();
        write!(node, "{}", text2).unwrap();
        assert_eq!(&node[..], [text, text2].concat().as_bytes());
        assert!(write!(node, "{}", text).is_err());
    }
}
