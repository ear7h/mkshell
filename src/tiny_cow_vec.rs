#[derive(Clone, Copy, Debug)]
pub enum TinyCowVec<'a, T, const N: usize> {
    Borrowed(&'a [T]),
    Owned([T; N], usize),
}

impl<T, const N: usize> TinyCowVec<'_, T, N> {
    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::Borrowed(s) => s,
            Self::Owned(a, n) => &a[..*n],
        }
    }
}

impl<T: PartialEq, const NL: usize, const NR: usize> PartialEq<TinyCowVec<'_, T, NR>>
    for TinyCowVec<'_, T, NL>
{
    fn eq(&self, other: &TinyCowVec<'_, T, NR>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq, const N: usize> Eq for TinyCowVec<'_, T, N> {}
