#![no_std]

use core::{mem::MaybeUninit, ptr, slice};
use generic_array::{typenum::marker_traits::Unsigned, ArrayLength, GenericArray};

pub mod typenum {
    pub use generic_array::typenum::consts;
}

pub trait DMANode {
    type Target;

    /// Creates a new node
    fn new() -> Self;

    /// Gives a `&mut [W]` slice to write into with the maximum size, the `commit` method
    /// must then be used to set the actual number of bytes written.
    ///
    /// Note that this function internally first zeros the non-initialized elements of the node's
    /// buffer.
    fn write(&mut self) -> &mut [Self::Target];

    /// Used to shrink the current size of the slice in the node, mostly used in conjunction
    /// with `write`.
    fn commit(&mut self, shrink_to: usize);

    /// Used to write data into the node, and returns how many bytes were written from `buf`.
    ///
    /// If the node is already partially filled, this will continue filling the node.
    fn write_slice(&mut self, buf: &[Self::Target]) -> usize;

    /// Clear the node of all data making it empty.
    fn clear(&mut self);

    /// Returns a readable slice which maps to the buffers internal data.
    fn read(&self) -> &[Self::Target];

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
    fn buffer_address_for_dma(&self) -> u32;

    /// Returns the maximum length of the internal buffer.
    fn max_len() -> usize;
}

pub struct Node<N, W>
where
    N: ArrayLength<MaybeUninit<W>> + Unsigned + 'static,
{
    len: usize,
    buf: GenericArray<MaybeUninit<W>, N>,
}

// Heavily inspired by @korken89 work
impl<N, W> DMANode for Node<N, W>
where
    N: ArrayLength<MaybeUninit<W>> + Unsigned + 'static,
{
    type Target = W;
    /// Creates a new node
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
            *elem = MaybeUninit::zeroed();
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
        let free = N::USIZE - self.len;
        let new_size = buf.len();
        let count = if new_size > free { free } else { new_size };

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

    fn clear(&mut self) {
        self.len = 0;
    }

    fn read(&self) -> &[W] {
        // Safe as it uses the internal length of valid data
        unsafe {
            slice::from_raw_parts(self.buf.as_slice().as_ptr() as *const _, self.len as usize)
        }
    }

    fn len(&self) -> usize {
        self.len as usize
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    unsafe fn set_len(&mut self, len: usize) {
        self.len = len;
    }

    fn buffer_address_for_dma(&self) -> u32 {
        self.buf.as_slice().as_ptr() as u32
    }

    fn max_len() -> usize {
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

#[cfg(test)]
mod tests {

    use crate::typenum::consts::U8;
    use crate::{DMANode, Node};
    use core::ptr;

    const DATA: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8];

    #[test]
    fn write_read() {
        let mut node = Node::<U8, u8>::new();
        let written = node.write_slice(DATA);
        assert_eq!(written, DATA.len());
        assert_eq!(node.len(), DATA.len());
        assert_eq!(node.read(), DATA);
        assert_eq!(Node::<U8, u8>::max_len(), 8);
    }

    #[test]
    fn write_with() {
        let mut node = Node::<U8, u8>::new();
        let max_len = core::cmp::min(Node::<U8, u8>::max_len(), DATA.len());
        unsafe {
            node.write_with(|buf, _len| {
                ptr::copy_nonoverlapping(DATA.as_ptr(), buf.as_mut_ptr() as *mut u8, max_len);
                max_len
            });
        }
        assert_eq!(node.read(), DATA);
    }
}