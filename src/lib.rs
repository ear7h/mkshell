#![allow(dead_code, unused_imports)]
#![warn(unused_features)]
#![feature(format_args_capture)]
#![feature(associated_type_bounds)]
#![feature(type_ascription)]
#![feature(generic_associated_types)]

use std::borrow::Borrow;
use std::convert::TryInto;

pub mod aho_corasick;
use crate::aho_corasick::{AhoCorasick, AhoCorasickBuilder};

pub mod tiny_cow_vec;
use tiny_cow_vec::TinyCowVec;

macro_rules! assert_trait {
    ($trait_name:tt, $value:expr) => {{
        fn assert_trait_fn<T: $trait_name>(_t: &T) {}
        let ret = $value;
        assert_trait_fn(&ret);
        ret
    }};
}

pub trait Cursor<'data> {
    type Item: 'data;

    fn get(&self) -> Self::Item;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn advance(&mut self, n: usize);
}

impl<'data> Cursor<'data> for &'data [u8] {
    type Item = &'data [u8];

    fn get(&self) -> &'data [u8] {
        self
    }

    fn len(&self) -> usize {
        (*self).len()
    }

    fn advance(&mut self, n: usize) {
        *self = &(*self)[n..];
    }
}

impl<'data> Cursor<'data> for &'data str {
    type Item = &'data str;

    fn get(&self) -> &'data str {
        self
    }

    fn len(&self) -> usize {
        (*self).len()
    }

    fn advance(&mut self, n: usize) {
        *self = &(*self)[n..];
    }
}

fn scan<'a, S, C>(s: &mut S, cursor: &mut C) -> S::Token<'a>
where
    S: Scanner,
    C: Cursor<'a, Item = S::Input<'a>>,
{
    let (ret, n) = s.scan(cursor.get());
    cursor.advance(n);
    ret
}

pub fn map<'a, C, S, F, T>(cursor: &mut C, scanner: &mut S, f: F) -> T
where
    C: Cursor<'a>,
    S: Scanner<Input<'a> = C::Item>,
    F: FnOnce(S::Token<'a>) -> T,
{
    let (tok, n) = scanner.scan(cursor.get());
    cursor.advance(n);
    f(tok)
}

pub fn map1<'a, C, S, F, T, P>(mut cursor: C, scanner: &mut S, f: F, p: P) -> T
where
    C: Cursor<'a>,
    S: Scanner<Input<'a> = C::Item>,
    F: FnOnce(P, S::Token<'a>) -> T,
{
    let (tok, n) = scanner.scan(cursor.get());
    cursor.advance(n);
    f(p, tok)
}

pub fn pipe<'a, 'b, C, C1, S1, S2>(
    cursor: &mut C,
    s1: &mut S1,
    s2: &'b mut S2,
) -> Vec<S2::Token<'a>>
where
    'b: 'a,
    C: Cursor<'a>,
    C1: Cursor<'a>,
    S1: Scanner<Input<'a> = C::Item, Token<'a> = C1>,
    S2: Scanner<Input<'a> = C1::Item, Token<'a> : 'a>,
{
    let (tok, n) = s1.scan(cursor.get());
    cursor.advance(n);
    s2.iter(tok).collect()
}

pub trait Scanner {
    type Input<'data>;
    type Token<'data>;

    fn scan<'a>(&mut self, b: Self::Input<'a>) -> (Self::Token<'a>, usize);

    fn iter<'a, C>(&mut self, cursor: C) -> Iter<&mut Self, C>
    where
        C: Cursor<'a, Item = Self::Input<'a>>,
        Self: Sized,
    {
        assert_trait!(
            Iterator,
            Iter {
                cursor,
                s: self,
                _marker: Default::default()
            }
        )
    }
}

pub struct Iter<'a, S, C> {
    s: S,
    cursor: C,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a, S, C> Iterator for Iter<'a, &mut S, C>
where
    S: Scanner,
    C: Cursor<'a, Item = S::Input<'a>>,
{
    type Item = S::Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor.is_empty() {
            return None;
        }

        let (t, n) = self.s.scan(self.cursor.get());
        self.cursor.advance(n);
        Some(t)
    }
}

pub struct CharSplit {
    c: char,
}

impl CharSplit {
    pub fn new(c: char) -> Self {
        CharSplit { c }
    }
}

impl Scanner for CharSplit {
    type Input<'a> = &'a str;
    type Token<'a> = &'a str;

    fn scan<'a>(&mut self, data: Self::Input<'a>) -> (Self::Token<'a>, usize) {
        for (idx, c) in data.char_indices() {
            if c == self.c {
                return (&data[..idx], idx + 1);
            }
        }

        (data, data.len())
    }
}

pub struct AnsiScanner {
    buf: [u8; AnsiScanner::PATTERN_MAX_SIZE],
    n: u8,
    buf_clear_n: u8,
    imm_match: bool,
    ac: AhoCorasick<u8, AnsiToken<'static>>,
}

impl Default for AnsiScanner {
    fn default() -> Self {
        Self::new()
    }
}


impl AnsiScanner {
    const PATTERNS: [(AnsiToken<'static>, &'static [u8]); 10] = [
        (AnsiToken::Up, &[0x1b, 0x5b, 0x41]),
        (AnsiToken::Down, &[0x1b, 0x5b, 0x42]),
        (AnsiToken::Right, &[0x1b, 0x5b, 0x43]),
        (AnsiToken::Left, &[0x1b, 0x5b, 0x44]),
        (AnsiToken::LeftWord, &[0x1b, 0x62]),
        (AnsiToken::RightWord, &[0x1b, 0x66]),
        (AnsiToken::Backspace, &[0x08]),
        (AnsiToken::Delete, &[0x1b, 0x5b, 0x33, 0x7e]),
        (AnsiToken::CtrlC, &[0x03]),
        (AnsiToken::CtrlD, &[0x04]),
    ];

    const PATTERN_MAX_SIZE: usize = 4;

    pub fn new() -> Self {
        let mut acb = AhoCorasickBuilder::default();
        for (val, key) in Self::PATTERNS.iter() {
            acb.insert(key.iter().cloned(), *val)
        }

        AnsiScanner {
            buf: [0; Self::PATTERN_MAX_SIZE],
            n: 0,
            buf_clear_n: 0,
            imm_match: false,
            ac: acb.build(),
        }
    }
}

impl Scanner for AnsiScanner {
    type Input<'a> = &'a [u8];
    type Token<'a> = AnsiToken<'a>;

    fn scan<'a>(&mut self, data: Self::Input<'a>) -> (Self::Token<'a>, usize) {
        if self.imm_match {
            self.imm_match = false;
            return (self.ac.matches().cloned().next().unwrap(), 0);
        }

        if self.n == 0 {
            // assert!(self.ac.match_buffer_size() == self.buf_clear_n.into());

            let start = self.buf_clear_n as usize;
            self.buf_clear_n = 0;

            for idx in start..data.len() {
                self.ac.push(&data[idx]);
                let mut m = self.ac.matches();

                if let Some(token) = m.next() {
                    // this scanner should never have more than one match at a time
                    assert!(m.next().is_none());

                    let n = self.ac.match_buffer_size();

                    return if n == idx + 1 {
                        (*token, idx + 1)
                    } else {
                        self.buf_clear_n = n.try_into().unwrap();
                        self.imm_match = true;
                        let ret_n = idx + 1 - n;
                        let ret = &data[start..ret_n];
                        (AnsiToken::Other(TinyCowVec::Borrowed(ret)), ret_n)
                    };
                }
            }

            let l = data.len();
            let n = self.ac.match_buffer_size();
            self.n = n.try_into().unwrap();
            self.buf[..n].copy_from_slice(&data[(l - n)..]);

            (
                AnsiToken::Other(TinyCowVec::Borrowed(&data[start..(l - n)])),
                l,
            )
        } else {
            assert!(self.ac.match_buffer_size() == self.n.into());

            while self.ac.match_buffer_size() > self.buf_clear_n as usize {
                let mut m = self.ac.push_matches(&data[self.buf_clear_n as usize]);
                self.buf_clear_n += 1;

                if let Some(token) = m.next() {
                    assert!(m.next().is_none());
                    return (*token, 0);
                }
            }

            let n = self.n;
            self.n = 0;
            let owned = self.buf;

            (AnsiToken::Other(TinyCowVec::Owned(owned, n as usize)), 0)
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AnsiToken<'a> {
    Up,
    Down,
    Right,
    Left,

    LeftWord,
    RightWord,

    Backspace,
    Delete,

    CtrlC,
    CtrlD,

    Other(TinyCowVec<'a, u8, { AnsiScanner::PATTERN_MAX_SIZE }>),
}

#[derive(Default)]
pub struct NewlineScanner {
    was_cr: bool,
}

impl NewlineScanner {
    pub fn new() -> Self {
        Self { was_cr: false }
    }
}

impl Scanner for NewlineScanner {
    type Input<'a> = &'a str;
    type Token<'a> = NewlineToken<'a>;

    fn scan<'a>(&mut self, b: Self::Input<'a>) -> (Self::Token<'a>, usize) {
        let mut it = b.char_indices();

        if self.was_cr {
            self.was_cr = false;

            match it.next() {
                Some((i, '\n')) => {
                    assert!(i == 0);
                    return (NewlineToken::Line(&b[..0]), 1);
                }
                Some((i, _)) => {
                    assert!(i == 0);
                    return (NewlineToken::Line(&b[..0]), 0);
                }
                None => return (NewlineToken::NonLine(&b[..0]), 0),
            }
        }

        while let Some((i, c)) = it.next() {
            if c == '\n' {
                return (NewlineToken::Line(&b[..i]), i + 1);
            }

            if c == '\r' {
                match it.next() {
                    Some((ii, '\n')) => return (NewlineToken::Line(&b[..i]), ii + 1),
                    Some(_) => return (NewlineToken::Line(&b[..i]), i + 1),
                    None => {
                        self.was_cr = true;
                        return (NewlineToken::NonLine(&b[..i]), i + 1);
                    }
                }
            }
        }

        let l = b.len();
        return (NewlineToken::NonLine(&b[..l]), l);
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum NewlineToken<'a> {
    Line(&'a str),
    NonLine(&'a str),
}

#[derive(Default)]
pub struct Utf8 {}

impl Utf8 {
    pub fn new() -> Self {
        Utf8 {}
    }
}

impl Scanner for Utf8 {
    type Input<'a> = &'a [u8];
    type Token<'a> = Utf8Token<'a>;

    fn scan<'a>(&mut self, b: Self::Input<'a>) -> (Self::Token<'a>, usize) {
        match std::str::from_utf8(b) {
            Ok(s) => (Utf8Token::Valid(s), s.len()),
            Err(e) => {
                let (valid, after_valid) = b.split_at(e.valid_up_to());
                if !valid.is_empty() {
                    let s = unsafe { std::str::from_utf8_unchecked(valid) };

                    (Utf8Token::Valid(s), s.len())
                } else {
                    (Utf8Token::Invalid(after_valid), after_valid.len())
                }
            }
        }
    }
}

pub enum Utf8Token<'a> {
    Valid(&'a str),
    Invalid(&'a [u8]),
}

pub fn utf8_replace<'a, 'b: 'a>(replacement_chars: &'b mut String, tok: Utf8Token<'a>) -> &'a str {
    use Utf8Token::*;

    match tok {
        Valid(s) => s,
        Invalid(b) => {
            let needs = b.len().saturating_sub(replacement_chars.len());
            if needs > 0 {
                replacement_chars.extend((0..needs).map(|_| char::REPLACEMENT_CHARACTER));
            }

            &replacement_chars[..b.len()]
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn newline_scanner() {
        use NewlineToken::*;

        let tc = vec![
            (&"a\nb", Line("a"), 2),
            (&"a\r\nb", Line("a"), 3),
            (&"\r\nb", Line(""), 2),
            (&"b", NonLine("b"), 1),
        ];

        for (input, exp_tok, exp_n) in tc {
            let mut scanner = NewlineScanner::new();
            let (got_tok, got_n) = scanner.scan(input);
            assert_eq!(got_tok, exp_tok);
            assert_eq!(got_n, exp_n);
        }

        {
            // test the bounds for iter()
            let mut scanner = NewlineScanner::new();
            let v = scanner.iter("test\ntest").collect(): Vec<_>;
            assert_eq!(v, vec![Line("test"), NonLine("test")]);
        }
    }

    #[test]
    fn ansi_scanner() {
        use AnsiToken::*;

        let tc = vec![
            (&b"\x1b\x5b\x41"[..], AnsiToken::Up, 3),
            (
                &b"abc\x1b\x5b\x41"[..],
                Other(TinyCowVec::Borrowed(b"abc")),
                3,
            ),
        ];

        for (input, exp_tok, exp_n) in tc {
            let mut scanner = AnsiScanner::new();
            let (got_tok, got_n) = scanner.scan(input);
            assert_eq!(got_tok, exp_tok);
            assert_eq!(got_n, exp_n);
        }

        {
            // test the bounds for iter()
            let mut scanner = AnsiScanner::new();
            let v = scanner.iter(&b"test\x08space"[..]).collect(): Vec<_>;
            assert_eq!(
                v,
                vec![
                    Other(TinyCowVec::Borrowed(b"test")),
                    Backspace,
                    Other(TinyCowVec::Borrowed(b"space")),
                ]
            );
        }
    }
}
