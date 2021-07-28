#![allow(dead_code)]

#![feature(format_args_capture)]
#![feature(associated_type_bounds)]
#![feature(type_ascription)]

mod aho_corasick;

use std::io;

trait Cursor<'data> {
    type Item : ?Sized;

    fn get(&self) -> &'data Self::Item;
    fn len(&self) -> usize;
    fn advance(&mut self, n : usize);
}

impl<'data> Cursor<'data> for &'data [u8] {
    type Item = [u8];

    fn get(&self) -> &'data [u8] {
        self
    }

    fn len(&self) -> usize {
        (*self).len()
    }

    fn advance(&mut self, n : usize) {
        *self = &(*self)[n..];
    }
}

impl <'data> Cursor<'data> for &'data str {
    type Item = str;

    fn get(&self) -> &'data str {
        self
    }

    fn len(&self) -> usize {
        (*self).len()
    }

    fn advance(&mut self, n : usize) {
        *self = &(*self)[n..];
    }
}

trait Scanner<'data> {
    type Input : ?Sized;
    type Token : 'data;

    fn scan(
        &mut self,
        b : &'data Self::Input
    ) -> (Self::Token, usize);

    fn iter<'scanner, 'cursor, C>(
        &'scanner mut self,
        cursor : &'cursor mut C, //&'cursor mut &'data Self::Input
    ) -> ScannerIter<'scanner, 'cursor, 'data, Self, C>
    where
        C : Cursor<'data, Item = Self::Input>,
        Self : Sized,
    {
        ScannerIter{
            cursor,
            s : self,
            _marker : Default::default()
        }
    }
}

struct ScannerIter<'scanner, 'cursor, 'data, S, C>
where
    S : Scanner<'data>,
    C : Cursor<'data, Item = S::Input>,
{
    s : &'scanner mut S,
    cursor : &'cursor mut C,
    _marker : std::marker::PhantomData<&'data ()>
}

impl<'data, S, C> Iterator for ScannerIter<'_, '_, 'data, S, C>
where
    S : Scanner<'data, Input : 'data>,
    C : Cursor<'data, Item = S::Input>,
{
    type Item = S::Token;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor.len() == 0 {
            return None
        }

        let (t, n) = self.s.scan(self.cursor.get());
        self.cursor.advance(n);
        Some(t)
    }
}


struct AnsiScanner{
}

impl AnsiScanner {
    const PATTERNS: [(AnsiToken<'static>, &'static [u8]); 10] = [
        (AnsiToken::Up,        &[0x1b, 0x5b, 0x41]),
        (AnsiToken::Down,      &[0x1b, 0x5b, 0x42]),
        (AnsiToken::Right,     &[0x1b, 0x5b, 0x43]),
        (AnsiToken::Left,      &[0x1b, 0x5b, 0x44]),
        (AnsiToken::LeftWord,  &[0x1b, 0x62]),
        (AnsiToken::RightWord, &[0x1b, 0x66]),
        (AnsiToken::Backspace, &[0x08]),
        (AnsiToken::Delete,    &[0x1b, 0x5b, 0x33, 0x7e]),
        (AnsiToken::CtrlC,     &[0x03]),
        (AnsiToken::CtrlD,     &[0x04]),
    ];

    fn new() -> Self {
        AnsiScanner{}
    }
}

impl<'a> Scanner<'a> for AnsiScanner {
    type Input = [u8];
    type Token = AnsiToken<'a>;

    fn scan(&mut self, b : &'a [u8]) -> (Self::Token, usize) {
        (AnsiToken::Other(b), b.len())
    }
}

#[derive(Debug)]
enum AnsiToken<'a> {
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

    Other(&'a [u8]), // everything else
}



struct NewlineScanner {
    was_cr : bool,
}

impl NewlineScanner {
    fn new() -> Self {
        Self{
            was_cr : false,
        }
    }
}

impl <'a> Scanner<'a> for NewlineScanner {
    type Input = str;
    type Token = NewlineToken<'a>;

    fn scan(&mut self, b : &'a str) -> (Self::Token, usize) {
        for (i, c) in b.char_indices() {
            if c == '\n' {
                let end = if self.was_cr && i > 0{
                    i - 1
                } else {
                    i
                };

                return (NewlineToken::Line(&b[..end]), i+1)
            }

            self.was_cr = c == '\r';
        }

        let mut l = b.len();

        if self.was_cr && l > 0 {
            l -= 1
        }

        return (NewlineToken::NonLine(&b[..l]), l)
    }
}

#[derive(Debug, PartialEq, Eq)]
enum NewlineToken<'a> {
    Line(&'a str),
    NonLine(&'a str),
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn ansi_scanner() {
        let s = "abc".to_string();
        let mut s1 = s.as_bytes();

        {
            let mut scanner = AnsiScanner::new();
            for tok in scanner.iter(&mut s1) {
                println!("{:?}", tok);
            }
        }

        println!("{:?}", s1.len());
    }

    #[test]
    fn newline_scanner() {
        use NewlineToken::*;

        let tc = vec![
            (
                &"a\nb",
                Line("a"),
                2,
            ),
            (
                &"a\r\nb",
                Line("a"),
                3,
            ),
            (
                &"\r\nb",
                Line(""),
                2,
            ),
            (
                &"b",
                NonLine("b"),
                1,
            ),
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
            let v = scanner.iter(&mut "test\ntest").collect() : Vec<_>;
            assert_eq!(v, vec![Line("test"), NonLine("test")]);
        }
    }
}
