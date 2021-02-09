#![warn(unused_features)]
#![feature(str_internals)]
#![feature(shrink_to)]

use aho_corasick::{self as ac, AhoCorasickBuilder};
use clap::{crate_version, App, Arg};
use core::str::utf8_char_width;
use mio::{unix::SourceFd, Events, Interest, Poll, Token};
use nix::fcntl::{fcntl, FcntlArg, OFlag};
use std::{
    fs::OpenOptions,
    io::{self, stdin, stdout, ErrorKind, Read, Write},
    os::unix::io::AsRawFd,
    process::{Child, Command, Stdio},
    str::from_utf8_unchecked,
};
use termion::{self, clear, cursor, raw::IntoRawMode};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

#[derive(Debug)]
enum Error {
    MissingStdin,
    MissingStdout,
    Io(io::Error),
    Nix(nix::Error),
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<nix::Error> for Error {
    fn from(e: nix::Error) -> Self {
        Error::Nix(e)
    }
}

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Default)]
struct Line {
    s: String,
}

impl Line {
    fn len(&self) -> usize {
        self.s.len()
    }

    fn as_str(&self) -> &str {
        self.s.as_str()
    }

    fn reset(&mut self) {
        self.s.truncate(0);
        if self.s.capacity() > 256 {
            self.s.shrink_to(256);
        }
    }

    fn push(&mut self, c: char) {
        self.s.push(c);
    }
}

struct LineBuf {
    lines: Vec<Line>,

    // oh yea that sweet 3 byte savings on padding
    overflow: [u8; 3],
    overflow_len: u8,

    start: usize,
    len: usize,
    chunk_size: usize,
}

impl std::fmt::Debug for LineBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[allow(dead_code)]
impl LineBuf {
    fn new(chunk_size: usize, nchunk: usize) -> LineBuf {
        let mut lines = Vec::new();
        lines.resize_with(nchunk.max(1), Default::default);

        LineBuf {
            lines,
            chunk_size,
            overflow: [0; 3],
            overflow_len: 0,
            start: 0,
            len: 1,
        }
    }

    fn get(&self, i: usize) -> Option<&Line> {
        if i >= self.len {
            None
        } else {
            let idx = (self.start + i) % self.lines.len();
            Some(&self.lines[idx])
        }
    }

    fn current_line(&mut self) -> &mut Line {
        let idx = (self.start + self.len - 1) % self.lines.len();
        &mut self.lines[idx]
    }

    fn new_line(&mut self) {
        if self.len < self.lines.len() {
            self.len += 1;
        } else {
            self.start = (self.start + 1) % self.lines.len();
            self.current_line().reset();
        }
    }

    fn push_str(&mut self, s: &str) {
        let chunk_size = self.chunk_size;
        let mut line = self.current_line();

        for c in s.chars() {
            line.push(c);

            if c == '\n' || c == '\r' || line.len() > chunk_size {
                self.new_line();
                line = self.current_line();
            }
        }
    }

    fn push_invalid_bytes(&mut self, b: &[u8]) {
        self.push_str("\u{fffd}".repeat(b.len()).as_str())
    }

    fn push(&mut self, c: char) {
        let mut b: [u8; 4] = [0; 4];
        self.push_str(c.encode_utf8(&mut b))
    }

    fn push_bytes(&mut self, data: &[u8]) {
        let mut buf = data;

        let overflow_len = self.overflow_len as usize;

        if overflow_len > 0 {
            let mut overflow: [u8; 4] = [0; 4];

            overflow[..overflow_len]
                .copy_from_slice(&self.overflow[..overflow_len]);

            let needs = (4 - overflow_len).min(data.len());

            overflow[overflow_len..(overflow_len + needs)]
                .copy_from_slice(&data[..needs]);

            let token = utf8_scan(&overflow[..(overflow_len + needs)]);
            //println!("{:?}", token);
            match token {
                Utf8Token::Empty => unreachable!(),
                Utf8Token::Incomplete(_) => {
                    assert!(data.len() < needs);
                    return;
                }
                Utf8Token::Normal(s) => {
                    self.push_str(s);

                    assert!(s.len() > overflow_len);

                    self.overflow_len = 0;
                    buf = &buf[needs..];
                }
                Utf8Token::Invalid(s) => {
                    self.push_invalid_bytes(s);

                    assert!(s.len() > overflow_len);

                    self.overflow_len = 0;
                    buf = &buf[needs..];
                }
            }
        }

        loop {
            let token = utf8_scan(buf);

            //println!("{:?}", token);

            match token {
                Utf8Token::Empty => break,
                Utf8Token::Incomplete(s) => {
                    assert!(buf.len() == s.len() && s.len() < 4);
                    self.overflow[..s.len()].copy_from_slice(s);
                    self.overflow_len = s.len() as u8;
                    break;
                }
                Utf8Token::Normal(s) => {
                    self.push_str(s);
                    buf = &buf[s.len()..];
                }
                Utf8Token::Invalid(b) => {
                    self.push_invalid_bytes(b);
                    buf = &buf[b.len()..];
                }
            }
        }
    }

    fn iter(&self) -> LineBufIter {
        LineBufIter {
            inner: self,
            start: 0,
            end: self.len,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Utf8Token<'a> {
    Empty,
    Normal(&'a str),
    Invalid(&'a [u8]),
    Incomplete(&'a [u8]),
}

fn utf8_scan(src: &[u8]) -> Utf8Token {
    if src.is_empty() {
        return Utf8Token::Empty;
    }

    let mut n = 0;
    let mut it = src.iter();
    let mut last = 0;

    macro_rules! next {
        () => {{
            n += 1;
            it.next()
        }};
    }

    macro_rules! must_next {
        () => {
            if let Some(b) = next!() {
                *b
            } else {
                return if last > 0 {
                    Utf8Token::Normal(unsafe {
                        from_utf8_unchecked(&src[..last])
                    })
                } else {
                    Utf8Token::Incomplete(&src[..n - 1])
                };
            }
        };
    }

    macro_rules! last_or_next_bound {
        () => {
            if last > 0 {
                Utf8Token::Normal(unsafe { from_utf8_unchecked(&src[..last]) })
            } else {
                while let Some(b) = next!() {
                    if is_char_boundary(*b) {
                        break;
                    }
                }
                Utf8Token::Invalid(&src[..n - 1])
            }
        };
    }

    loop {
        let b = if let Some(b) = next!() {
            *b
        } else {
            return Utf8Token::Normal(unsafe {
                from_utf8_unchecked(&src[..n - 1])
            });
        };

        if b < 128 {
            last = n;
            continue;
        }

        let w = utf8_char_width(b);

        match w {
            0 => {
                // too long
                return last_or_next_bound!();
            }
            1 => unreachable!(), // handled earlier
            2 => {
                if !is_cont_byte(must_next!()) {
                    return last_or_next_bound!();
                }
            }
            3 => {
                match (b, must_next!()) {
                    (0xE0, 0xA0..=0xBF) => (),
                    (0xE1..=0xEC, 0x80..=0xBF) => (),
                    (0xED, 0x80..=0x9F) => (),
                    (0xEE..=0xEF, 0x80..=0xBF) => (),
                    _ => {
                        return last_or_next_bound!();
                    }
                }

                if !is_cont_byte(must_next!()) {
                    return last_or_next_bound!();
                }
            }
            4 => {
                match (b, must_next!()) {
                    (0xF0, 0x90..=0xBF) => (),
                    (0xF1..=0xF3, 0x80..=0xBF) => (),
                    (0xF4, 0x80..=0x8F) => (),
                    _ => {
                        return last_or_next_bound!();
                    }
                }

                if !is_cont_byte(must_next!()) {
                    return last_or_next_bound!();
                }

                if !is_cont_byte(must_next!()) {
                    return last_or_next_bound!();
                }
            }
            _ => unreachable!(),
        }
        last = n;
    }
}

fn is_char_boundary(b: u8) -> bool {
    (b as i8) >= -0x40
}

fn is_cont_byte(b: u8) -> bool {
    b >> 6 == 0b10
}

#[derive(Debug)]
enum LineToken {
    Empty,
    Wrap,
    Newline,
    Incomplete,
}

/// Tokenize the line from right to left, returns the token type, the index of the token
/// and the display width of the input between tokens
fn rline_scan(src: &str, line_width: usize) -> (LineToken, usize, usize) {
    if src.is_empty() {
        return (LineToken::Empty, 0, 0);
    }

    let mut w = 0;

    for (idx, c) in src.char_indices().rev() {
        match c {
            '\n' | '\r' => return (LineToken::Newline, idx, w),
            c => {
                let cw = c.width().unwrap_or(0);

                if w + cw > line_width {
                    return (LineToken::Wrap, idx, w);
                }

                w += cw;
            }
        }
    }

    (LineToken::Incomplete, 0, w)
}

/// AnsiToken and associated functions are not a full ansi parser. The intention
/// is for them to capture only the relevant KEY PRESSES to our functionality.
#[derive(Clone, Copy, Debug)]
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

/// AnsiScanner is a happy-path ANSI terminal code parser. It simply matches
/// against literals that represent `AnsiToken`s.
struct AnsiScanner<R> {
    i: ac::StreamChunkIter<R, u16>,
}

impl<R: Read> AnsiScanner<R> {
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

    fn new(r: R) -> Self {
        let a = AhoCorasickBuilder::new()
            .dfa(true)
            .prefilter(true)
            .premultiply(true)
            .build_with_size(Self::PATTERNS.iter().map(|x| x.1))
            .unwrap();

        AnsiScanner {
            i: ac::StreamChunkIter::new(a, r),
        }
    }

    fn next<'a>(&'a mut self) -> Option<Result<AnsiToken<'a>>> {
        let next = self.i.next();

        let chunk = match next {
            None => return None,
            Some(Err(e)) => return Some(Err(e.into())),
            Some(Ok(x)) => x,
        };

        match chunk {
            ac::StreamChunk::NonMatch { bytes, .. } => {
                Some(Ok(AnsiToken::Other(bytes)))
            }
            ac::StreamChunk::Match { mat, .. } => {
                Some(Ok(Self::PATTERNS[mat.pattern()].0))
            }
        }
    }
}

struct LineBufIter<'a> {
    inner: &'a LineBuf,
    start: usize,
    end: usize,
}

impl<'a> Iterator for LineBufIter<'a> {
    type Item = &'a Line;

    fn next(&mut self) -> Option<&'a Line> {
        if self.start >= self.end {
            None
        } else {
            let start = self.start;
            self.start += 1;
            self.inner.get(start)
        }
    }
}

impl<'a> DoubleEndedIterator for LineBufIter<'a> {
    fn next_back(&mut self) -> Option<&'a Line> {
        if self.start >= self.end {
            None
        } else {
            self.end -= 1;
            self.inner.get(self.end)
        }
    }
}

struct CursorMut<'a> {
    shell: &'a mut Shell,
}

impl CursorMut<'_> {
    fn cursor(&self) -> usize {
        self.shell.cursor
    }

    fn cursor_mut(&mut self) -> &mut usize {
        &mut self.shell.cursor
    }

    fn current(&self) -> &str {
        self.shell.history.current()
    }

    fn current_mut(&mut self) -> &mut String {
        self.shell.history.current_mut()
    }

    fn is_at_end(&self) -> bool {
        self.cursor() == self.current().len()
    }

    fn is_at_begining(&self) -> bool {
        self.cursor() == 0
    }

    fn right_char(&mut self) {
        let mut it = self.current()[self.cursor()..].chars();
        if let Some(c) = it.next() {
            *self.cursor_mut() += c.len_utf8();
        }
    }

    fn left_char(&mut self) {
        let mut it = self.current()[..self.cursor()].char_indices().rev();
        if let Some((idx, _)) = it.next() {
            *self.cursor_mut() = idx;
        }
    }

    fn right_word(&mut self) {
        let mut it = self.current()[self.cursor()..].char_indices().fuse();
        for (_, c) in &mut it {
            if !c.is_alphanumeric() {
                break;
            }
        }

        for (idx, c) in it {
            if c.is_alphanumeric() {
                *self.cursor_mut() += idx;
                return;
            }
        }

        *self.cursor_mut() = self.current().len();
    }

    fn left_word(&mut self) {
        // algo:
        // move to previous char
        // if prev is not alphanum
        //      move back to first alphanum
        // move back to first alphanum

        let mut it =
            self.current()[..self.cursor()].char_indices().rev().fuse();

        let mut last_idx;
        match it.next() {
            None => return,
            Some((idx, c)) if c.is_alphanumeric() => last_idx = idx,
            Some((idx, _)) => {
                last_idx = idx;
                for (idx, c) in &mut it {
                    last_idx = idx;
                    if c.is_alphanumeric() {
                        break;
                    }
                }
            }
        };

        for (idx, c) in it {
            if !c.is_alphanumeric() {
                break;
            }
            last_idx = idx;
        }

        *self.cursor_mut() = last_idx;
    }

    fn remove_char(&mut self) {
        let cursor = self.cursor();
        self.current_mut().remove(cursor);
    }
}

#[derive(Debug, Default)]
struct History {
    history: Vec<String>,
    edits: Vec<(usize, String)>,
    idx: usize,
}

impl History {
    fn new() -> Self {
        History {
            history: Vec::new(),
            edits: vec![(0, String::new())],
            idx: 0,
        }
    }

    // TODO: return some impl AsMut<Target=String> that
    // sets the edited flag to true on destruction after
    // testing for equality?
    fn current_mut(&mut self) -> &mut String {
        let edit_idx = self
            .edits
            .iter()
            .enumerate()
            .find(|x| ((x.1).0) == self.idx);

        assert!(self.idx < self.history.len() || edit_idx.is_some());

        let idx = if let Some((idx, _)) = edit_idx {
            idx
        } else {
            let c = self.history[self.idx].trim_end().to_string();
            self.edits.push((self.idx, c));
            self.edits.len() - 1
        };

        &mut self.edits[idx].1
    }

    fn current(&self) -> &str {
        let edit = self.edits.iter().find(|edit| edit.0 == self.idx);

        if let Some(edit) = edit {
            edit.1.as_str()
        } else {
            self.history[self.idx].as_str().trim_end()
        }
    }

    fn back(&mut self) {
        if self.idx > 0 {
            self.idx -= 1;
        }
    }

    fn forward(&mut self) {
        if self.idx < self.history.len() {
            self.idx += 1;
        }
    }

    /// sends the current to the history and clears any edits
    /// made since the last push()
    fn push(&mut self) {
        let c = self.current();
        if Some(c) != self.history.last().map(String::as_str) {
            let c = c.to_string();
            self.history.push(c);
        }

        self.idx = self.history.len();

        self.edits.truncate(1);
        self.edits[0].0 = self.history.len();
        self.edits[0].1.truncate(0);
    }
}

#[derive(Debug)]
struct Shell {
    o: LineBuf,
    history: History,
    can_send: bool,
    cursor: usize,
}

impl Shell {
    fn new(mut cols: usize, mut rows: usize) -> Shell {
        if cols == 0 {
            cols = 80
        }

        if rows == 0 {
            rows = 24
        }

        Shell {
            o: LineBuf::new(cols, rows),
            history: History::new(),
            can_send: false,
            cursor: 0,
        }
    }

    fn history(&mut self) -> &mut History {
        &mut self.history
    }

    fn cursor(&mut self) -> CursorMut {
        CursorMut { shell: self }
    }

    fn push_in_slice(&mut self, b: &[u8]) {
        // TODO: use u8 buffers in the input and history, convert unreadable
        // characters to u+fffd at render time
        let s = String::from_utf8_lossy(b);
        for c in s.chars() {
            self.push_in(c)
        }
    }

    fn push_in(&mut self, c: char) {
        let input = self.history.current_mut();

        // I think '\r' are getting sent via raw reads from stdin
        let c = if c == '\r' { '\n' } else { c };

        if c == '\n' {
            if self.cursor < input.len() && self.cursor != 0 {
                input.insert(self.cursor, '\n');
                self.cursor += '\n'.len_utf8();
            } else if input.ends_with('\\') {
                input.pop();
                input.push(c);
                self.cursor = input.len();
            } else {
                input.push('\n');
                self.cursor += c.len_utf8();
                self.cursor = self.cursor.min(input.len());
                self.can_send = true;
            }
        } else {
            self.cursor = self.cursor.min(input.len());
            input.insert(self.cursor, c);
            self.cursor += c.len_utf8();
        }
    }

    fn reset_in(&mut self) {
        self.history.push();
        self.can_send = false;
        self.cursor = 0;
    }

    fn poll(&self) -> Option<&str> {
        if self.can_send {
            Some(self.history.current())
        } else {
            None
        }
    }

    fn push_out_slice(&mut self, b: &[u8]) {
        self.o.push_bytes(b);
    }

    fn render<T: Term>(&self, term: &mut T) -> Result<()> {
        let (cols, rows) = term.size()?;
        if cols == 0 || rows == 0 {
            return Ok(());
        }

        term.cursor_off()?;
        term.clear()?;

        let mut render_row = rows - 1;

        let mut cursor_col = 0;
        let mut cursor_row = rows - 1;

        macro_rules! done {
            () => {{
                term.goto(cursor_col, cursor_row)?;
                term.cursor_on()?;
                return Ok(());
            }};
        }

        macro_rules! next_row {
            () => {{
                if render_row == 0 {
                    done!()
                }
                render_row -= 1;
                let _ = render_row; // get rid of unused_assignment lint
            }};
        }

        let current = self.history.current();
        let cursor = self.cursor.min(current.len());

        let mut nbytes = current.len();

        for (row_idx, line) in current.rsplit('\n').enumerate() {
            term.write_at(0, render_row, line)?;

            if (nbytes - line.len()..=nbytes).contains(&cursor) {
                let end = cursor - (nbytes - line.len());

                // the cursor can only past the end of the line
                // if the line has a newline at the end
                assert!(end <= line.len() || row_idx > 0);

                if end > line.len() {
                    cursor_row = rows - row_idx - 2;
                } else {
                    cursor_col = line[..end].width();
                    cursor_row = rows - row_idx - 1;
                }
            }

            // this should only saturate once, on the last line
            nbytes = nbytes.saturating_sub(line.len() + 1);

            next_row!();
        }

        term.write_at(0, render_row, "\u{2500}".repeat(cols).as_str())?;
        next_row!();

        let mut incompletes: Vec<(&str, usize)> = Vec::new();
        let mut incompletes_w = 0;
        let mut first_newline = true;

        for line in self.o.iter().rev().skip(1) {
            let mut line_str = line.as_str();

            loop {
                let token = rline_scan(&line_str, cols - incompletes_w);
                //println!("{:?}", token);
                match token {
                    (LineToken::Empty, _, _) => break,
                    (LineToken::Wrap, idx, w) => {
                        term.write_at(0, render_row, &line_str[idx + 1..])?;

                        let mut ww = w;
                        for (s, sw) in incompletes.iter().rev() {
                            term.write_at(ww, render_row, s)?;
                            ww += sw;
                        }

                        next_row!();
                        incompletes.truncate(0);
                        incompletes_w = 0;

                        line_str = &line_str[..=idx];
                    }
                    (LineToken::Newline, idx, w) => {
                        if first_newline {
                            line_str = &line_str[..idx];
                            first_newline = false;
                            continue;
                        }

                        term.write_at(0, render_row, &line_str[idx + 1..])?;

                        let mut ww = w;
                        for (s, sw) in incompletes.iter().rev() {
                            term.write_at(ww, render_row, s)?;
                            ww += sw;
                        }

                        next_row!();
                        incompletes.truncate(0);
                        incompletes_w = 0;

                        line_str = &line_str[..idx];
                    }
                    (LineToken::Incomplete, _, w) => {
                        incompletes.push((&line_str, w));
                        incompletes_w += w;
                        break;
                    }
                }
            }
        }

        if !incompletes.is_empty() {
            let mut ww = 0;
            for (s, sw) in incompletes.iter().rev() {
                term.write_at(ww, render_row, s)?;
                ww += sw;
            }

            next_row!();
        }

        done!()
    }
}

trait Term {
    // (cols, rows)
    fn size(&mut self) -> Result<(usize, usize)>;

    // 0-indexed
    fn write_at(&mut self, x: usize, y: usize, s: &str) -> Result<()>;
    fn goto(&mut self, x: usize, y: usize) -> Result<()>;

    fn clear(&mut self) -> Result<()>;

    fn cursor_on(&mut self) -> Result<()>;
    fn cursor_off(&mut self) -> Result<()>;

    fn term_flush(&mut self) -> Result<()>;
}

struct TestTerm<W> {
    size: (usize, usize),
    o: W,
}

impl<W: Write> Term for TestTerm<W> {
    // (cols, rows)
    fn size(&mut self) -> Result<(usize, usize)> {
        write!(self.o, "size() -> {:?}\r\n", self.size)?;
        Ok(self.size)
    }

    // 0-indexed
    fn write_at(&mut self, x: usize, y: usize, s: &str) -> Result<()> {
        write!(
            self.o,
            "write_at(x : {}, y : {}, s : {:?}) -> Ok(())\r\n",
            x, y, s
        )?;
        Ok(())
    }

    fn goto(&mut self, x: usize, y: usize) -> Result<()> {
        write!(self.o, "goto(x : {}, y : {}) -> Ok(())\r\n", x, y)?;
        Ok(())
    }

    fn clear(&mut self) -> Result<()> {
        write!(self.o, "clear() -> Ok(())\r\n")?;
        Ok(())
    }

    fn cursor_on(&mut self) -> Result<()> {
        write!(self.o, "cursor_on() -> Ok(())\r\n")?;
        Ok(())
    }

    fn cursor_off(&mut self) -> Result<()> {
        write!(self.o, "cursor_off() -> Ok(())\r\n")?;
        Ok(())
    }

    fn term_flush(&mut self) -> Result<()> {
        write!(self.o, "flush() -> Ok(())\r\n")?;
        Ok(())
    }
}

impl<W: Write> Term for W {
    fn size(&mut self) -> Result<(usize, usize)> {
        match termion::terminal_size() {
            Ok((c, r)) => Ok((c as usize, r as usize)),
            Err(e) => Err(e.into()),
        }
    }

    fn write_at(&mut self, x: usize, y: usize, s: &str) -> Result<()> {
        write!(
            self,
            "{}{}",
            cursor::Goto((x + 1) as u16, (y + 1) as u16),
            s
        )?;
        Ok(())
    }

    fn goto(&mut self, x: usize, y: usize) -> Result<()> {
        write!(self, "{}", cursor::Goto((x + 1) as u16, (y + 1) as u16))?;
        Ok(())
    }

    fn clear(&mut self) -> Result<()> {
        write!(self, "{}", clear::All)?;
        Ok(())
    }

    fn cursor_on(&mut self) -> Result<()> {
        write!(self, "{}", cursor::Show)?;
        Ok(())
    }

    fn cursor_off(&mut self) -> Result<()> {
        write!(self, "{}", cursor::Hide)?;
        Ok(())
    }

    fn term_flush(&mut self) -> Result<()> {
        std::io::Write::flush(self)?;
        Ok(())
    }
}

/// Returns the file flags
#[allow(dead_code)]
fn is_non_blocking<F: AsRawFd>(f: &F) -> OFlag {
    let fd = f.as_raw_fd();
    println!("fd {}", fd);
    let flag = fcntl(fd, FcntlArg::F_GETFL).unwrap();
    OFlag::from_bits(flag).unwrap()
}

/// Makes the fd that backs f non-blocking by calling
/// fcntl with the O_NONBLOCK flag
fn make_non_blocking<F: AsRawFd>(f: &F) -> Result<()> {
    let fd = f.as_raw_fd();
    let flag = fcntl(fd, FcntlArg::F_GETFL).unwrap();
    fcntl(
        fd,
        FcntlArg::F_SETFL(OFlag::from_bits(flag).unwrap() | OFlag::O_NONBLOCK),
    )?;
    Ok(())
}

/// use a process as a pipe, writing all of r into stdin (consuming r) and
/// writing all the output of p into w (consuming p)
fn proc_pipe<R, W>(mut p: Child, mut r: R, w: &mut W) -> Result<()>
where
    R: Read,
    W: Write,
{
    const TOKEN_PROC_IN: mio::Token = Token(0);
    const TOKEN_PROC_OUT: mio::Token = Token(1);

    // using a macro here because this call needs to be made multiple times
    // but eventually the inner p.stdin.take() object gets dropped in order
    // to close the fd (so the process can exit), and using a local var
    // causes a compile error (cannot consume proc_in while there is also
    // a reference)
    macro_rules! proc_in {
        () => {
            p.stdin.as_mut().ok_or(Error::MissingStdin)?
        };
    }

    macro_rules! proc_out {
        () => {
            p.stdout.as_mut().ok_or(Error::MissingStdout)?;
        };
    }

    let mut poll = Poll::new()?;

    poll.registry().register(
        &mut SourceFd(&proc_in!().as_raw_fd()),
        TOKEN_PROC_IN,
        Interest::WRITABLE,
    )?;

    poll.registry().register(
        &mut SourceFd(&proc_out!().as_raw_fd()),
        TOKEN_PROC_OUT,
        Interest::READABLE,
    )?;

    let mut events = Events::with_capacity(2);

    let mut buf = [0; 1024];

    loop {
        poll.poll(&mut events, None)?;

        for event in events.iter() {
            match event.token() {
                TOKEN_PROC_IN => {
                    let n = r.read(&mut buf)?;
                    if n == 0 {
                        if let Some(x) = p.stdin.take() {
                            std::mem::drop(x);
                            continue;
                        }
                    }

                    match proc_in!().write_all(&buf[..n]) {
                        Ok(_) => {}
                        Err(err)
                            if matches!(
                                err.kind(),
                                ErrorKind::WouldBlock | ErrorKind::Interrupted
                            ) =>
                        {
                            break
                        }
                        Err(err) => return Err(err.into()),
                    }
                }
                TOKEN_PROC_OUT => {
                    match proc_out!().read(&mut buf) {
                        Ok(n) => {
                            if n == 0 {
                                return Ok(());
                            }

                            w.write_all(&buf[..n])?;
                        }
                        Err(err)
                            if matches!(
                                err.kind(),
                                ErrorKind::WouldBlock | ErrorKind::Interrupted
                            ) =>
                        {
                            continue
                        }
                        Err(err) => return Err(err.into()),
                    };
                }
                _ => {}
            }
        }
    }
}

fn main_loop<I, O>(
    stdin: I,
    mut stdout: O,
    mut child: Child,
    mut transformer: Option<Command>,
) -> Result<()>
where
    I: Read + AsRawFd,
    O: Write + AsRawFd,
{
    const TOKEN_STDIN: mio::Token = Token(0);
    const TOKEN_CHILD_STDOUT: mio::Token = Token(1);

    macro_rules! child_in {
        () => {
            child.stdin.as_mut().ok_or(Error::MissingStdin)?
        };
    }

    macro_rules! child_out {
        () => {
            child.stdout.as_mut().ok_or(Error::MissingStdout)?;
        };
    }

    let mut poll = Poll::new()?;

    // uncomment the following lines to check if stdout is being effected by
    // the fcntl call to stdin
    // println!("is_non_blocking {:?}", is_non_blocking(&term));

    make_non_blocking(&stdin)?;

    // println!("is_non_blocking {:?}", is_non_blocking(&term));

    poll.registry()
        .register(
            &mut SourceFd(&stdin.as_raw_fd()),
            TOKEN_STDIN,
            Interest::READABLE,
        )
        .unwrap();

    make_non_blocking(child_out!())?;

    poll.registry().register(
        &mut SourceFd(&child_out!().as_raw_fd()),
        TOKEN_CHILD_STDOUT,
        Interest::READABLE,
    )?;

    let mut events = Events::with_capacity(2);

    let mut shell = {
        let (cols, rows) = termion::terminal_size()?;
        if cols * rows == 0 {
            // panic!("cols : {}, rows : {}", cols, rows);
        }
        Shell::new(cols as usize, rows as usize)
    };

    let mut ansi_scanner = AnsiScanner::new(stdin);

    let mut buf = [0; 4 * (1 << 10)];
    let mut term_buf = Vec::<u8>::new();

    loop {
        poll.poll(&mut events, None)?;

        for event in events.iter() {
            // print!("event {:?}\r\n", event);
            match event.token() {
                TOKEN_STDIN => {
                    // read from stdin and push to terminal event parser
                    loop {
                        let token = match ansi_scanner.next() {
                            None => return Ok(()), // eof
                            Some(Ok(token)) => token,
                            Some(Err(Error::Io(err)))
                                if err.kind() == ErrorKind::WouldBlock =>
                            {
                                break
                            }
                            Some(Err(Error::Io(err)))
                                if err.kind() == ErrorKind::Interrupted =>
                            {
                                continue
                            }
                            Some(Err(err)) => return Err(err),
                        };

                        match token {
                            AnsiToken::Other(b) => {
                                shell.push_in_slice(b);
                                if let Some(s) = shell.poll() {
                                    // this is a &mut Process but spawn does
                                    // not change the way which future spawn
                                    // calls behave, see std::process::Command
                                    if let Some(ref mut cmd) = transformer {
                                        let cmd_run = cmd.spawn()?;

                                        proc_pipe(
                                            cmd_run,
                                            s.as_bytes(),
                                            child_in!(),
                                        )?;
                                    } else {
                                        child_in!().write_all(s.as_bytes())?;
                                    }

                                    child_in!().flush()?;
                                    shell.reset_in();
                                }
                            }
                            AnsiToken::Backspace => {
                                let mut c = shell.cursor();
                                if !c.is_at_begining() {
                                    c.left_char();
                                    c.remove_char();
                                }
                            }
                            AnsiToken::Delete => {
                                let mut c = shell.cursor();
                                if !c.is_at_end() {
                                    c.remove_char();
                                }
                            }
                            AnsiToken::Right => shell.cursor().right_char(),
                            AnsiToken::Left => shell.cursor().left_char(),
                            AnsiToken::RightWord => shell.cursor().right_word(),
                            AnsiToken::LeftWord => shell.cursor().left_word(),
                            AnsiToken::Up => {
                                shell.history().back();
                            }
                            AnsiToken::Down => {
                                shell.history().forward();
                            }

                            AnsiToken::CtrlD | AnsiToken::CtrlC => {
                                return Ok(())
                            }
                        }

                        term_buf.truncate(0);
                        shell.render(&mut term_buf)?;
                        stdout.write_all(term_buf.as_slice())?;
                        stdout.flush()?;
                    }
                }
                TOKEN_CHILD_STDOUT => {
                    // read from child stdout and push to shell out
                    match child_out!().read(&mut buf) {
                        Ok(0) => return Ok(()),
                        Ok(n) => {
                            shell.push_out_slice(&buf[..n]);

                            term_buf.truncate(0);
                            shell.render(&mut term_buf)?;
                            stdout.write_all(term_buf.as_slice())?;
                            stdout.flush()?;
                        }
                        Err(err) => {
                            match err.kind() {
                                ErrorKind::WouldBlock
                                | ErrorKind::Interrupted => continue,
                                _ => {
                                    //println!("{}:{} error scanning {:?}", file!(), line!(), err);
                                    return Err(err.into());
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

const ARG_INPUT_DECORATOR: &str = "arg-input-decorator";
const ARG_CMD: &str = "cmd";

fn main() {
    let matches = App::new("mkshell")
        .version(crate_version!())
        .arg(
            Arg::with_name(ARG_INPUT_DECORATOR)
                .long("input-decorator")
                .short("d")
                .value_name("SCRIPT")
                .help(
                    "SCRIPT will be piped user input and its output piped to \
                  CMD",
                ),
        )
        .arg(
            Arg::with_name(ARG_CMD)
                .required(true)
                .multiple(true)
                .value_name("CMD")
                .help(
                    "mkshell will create an interactive shell that writes to \
                and reads from CMD",
                ),
        )
        .get_matches();

    let cmd = matches.values_of(ARG_CMD).unwrap().collect::<Vec<_>>();

    let child = Command::new(cmd[0])
        .args(&cmd[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let decor = matches.value_of(ARG_INPUT_DECORATOR).map(|decor| {
        let arr = decor.split_ascii_whitespace().collect::<Vec<_>>();
        let mut c = Command::new(arr[0]);
        c.args(&arr[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        c
    });

    // for the main_loop the reads are non-blocking and writes are blocking
    // this allows the main_loop use only a single thread when doing the
    // data transfering (benefit of async) while also not having to create
    // a complicated job queue or scheduler. This means stdin needs to be
    // non-blocking while stdout is blocking. However, if fd 0 (stdin)
    // gets set to O_NONBLOCK using fcntl, then fd 1 (stdout) also
    // gets set to O_NONBLOCK for some unknown unix reason.
    //
    // https://stackoverflow.com/a/23866059
    // https://users.rust-lang.org/t/how-can-i-open-stdin-nonblocking-while-leaving-stdout-blocking/17635
    //
    // in order to get around this we need to open another file pointing to
    // stdin. I also tried "/dev/fd/1" but it did not work. Openeing /dev/tty
    // will probably have some unwanted side effects...
    //
    // so far the only side effect is that writing to /dev/tty is a lot slower
    let keep_raw = stdout().into_raw_mode().unwrap();
    let stdout2 = OpenOptions::new().write(true).open("/dev/tty").unwrap();

    let res = main_loop(stdin(), stdout2, child, decor);

    std::mem::drop(keep_raw);

    match res {
        Ok(_) => println!("\nbye!"),
        Err(err) => println!("\n{:?}", err),
    };
}
