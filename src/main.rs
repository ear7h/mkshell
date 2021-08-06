#![warn(unused_features)]
#![feature(type_ascription)]

use std::io::{self, Write};
use std::os::unix::io::AsRawFd;
use std::process::Stdio;
use std::rc::Rc;

use clap::{crate_version, App, Arg};
use nix::sys::termios;
use termion::{clear, cursor};
use tokio::io::{AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::Mutex;

use mkshell::Scanner;

#[macro_use]
extern crate quick_from;

const ARG_INPUT_DECORATOR: &str = "arg-input-decorator";
const ARG_CMD: &str = "cmd";

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, QuickFrom)]
pub enum Error {
    #[quick_from]
    Io(io::Error),
}

pub struct Ui {
    out: RawStdout,
    out_buf: Vec<u8>,
    in_buf: Vec<u8>,
    buf: Vec<u8>,
}

macro_rules! must_write {
    ($($arg:expr),*) => {
        write!(
            $(
                $arg ,
            )*
        ).unwrap();
    }
}

impl Ui {
    pub fn new(out: RawStdout) -> Self {
        Self {
            out,
            in_buf: Vec::new(),
            out_buf: Vec::new(),
            buf: Vec::new(),
        }
    }

    pub async fn draw(&mut self) -> Result<()> {
        self.buf.clear();

        must_write!(self.buf, "{}", cursor::Hide);
        must_write!(self.buf, "{}", cursor::Goto(1, 1));
        must_write!(self.buf, "{}", clear::All);

        let mut replacement_chars = String::new();

        must_write!(self.buf, "{}", "out: \r\n");
        for tok in mkshell::Utf8::new().iter(self.out_buf.as_slice()) {
            let s = mkshell::utf8_replace(&mut replacement_chars, tok);

            for tok in mkshell::NewlineScanner::new().iter(s) {
                use mkshell::NewlineToken::*;
                match tok {
                    Line(s) => must_write!(self.buf, "{}\r\n", s),
                    NonLine(s) => must_write!(self.buf, "{}", s),
                }
            }
        }

        must_write!(self.buf, "{}", "in: \r\n> ");
        for tok in mkshell::Utf8::new().iter(self.in_buf.as_slice()) {
            let s = mkshell::utf8_replace(&mut replacement_chars, tok);

            for tok in mkshell::NewlineScanner::new().iter(s) {
                use mkshell::NewlineToken::*;
                match tok {
                    Line(s) => must_write!(self.buf, "{}\r\n> ", s),
                    NonLine(s) => must_write!(self.buf, "{}", s),
                }
            }
        }

        must_write!(self.buf, "{}", cursor::Show);

        self.out.write_all(&self.buf).await?;
        self.out.flush().await?;
        Ok(())
    }

    pub fn push_output(&mut self, s: &[u8]) {
        self.out_buf.extend_from_slice(s);
    }

    pub fn push_input(&mut self, s: &[u8]) {
        self.in_buf.extend_from_slice(s);
    }

    pub fn get_input(&self) -> &[u8] {
        &self.in_buf
    }

    pub fn clear_input(&mut self) {
        self.in_buf.clear()
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    main1().await;
}

async fn main1() {
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
                .last(true)
                .value_name("CMD")
                .help(
                    "mkshell will create an interactive shell that writes to \
                and reads from CMD",
                ),
        )
        .get_matches();

    let cmd = matches.values_of(ARG_CMD).unwrap().collect::<Vec<_>>();
    let mut child = tokio::process::Command::new(cmd[0])
        .args(&cmd[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    let mut child_in = child.stdin.take().unwrap();

    let mut child_out = child.stdout.take().unwrap();
    let mut child_err = child.stderr.take().unwrap();

    let ui_src = Rc::new(Mutex::new(Ui::new(
        RawStdout::new(tokio::io::stdout()).unwrap(),
    )));

    let ui = Rc::clone(&ui_src);

    let stdin2proc = async {
        let mut stdin = tokio::io::stdin();

        let mut ansi = mkshell::AnsiScanner::new();

        let mut buf: [u8; 4096] = [0; 4096];

        loop {
            let n = stdin.read(&mut buf).await?;
            if n == 0 {
                return Ok(()): std::io::Result<()>;
            }

            let mut ui = ui.lock().await;

            for token in ansi.iter(&buf[..n]) {
                use mkshell::AnsiToken::*;
                match token {
                    CtrlC | CtrlD => {
                        // TODO: actually close the pipe and kill the proc
                        return Ok(()): std::io::Result<()>;
                    }
                    Down => {
                        ui.push_input(b"\r\n");
                    }
                    Other(s) => match s.as_slice() {
                        b"\n" | b"\r" => {
                            ui.push_input(b"\r\n");
                            child_in.write(ui.get_input()).await?;
                            ui.clear_input();
                            child_in.flush().await?;
                        }
                        s => {
                            ui.push_input(s);
                        }
                    },
                    _ => {}
                }
            }

            ui.draw().await.unwrap();
        }
    };

    let ui = Rc::clone(&ui_src);

    let proc2stdout = async {
        // let mut child_all = SelectReader::new(child_out, child_err);
        let mut buf_out: [u8; 4096] = [0; 4096];
        let mut buf_err: [u8; 4096] = [0; 4096];

        loop {
            let buf = tokio::select! {
                res = child_out.read(&mut buf_out) => {
                    let n = res.unwrap();
                    &buf_out[..n]
                },
                res = child_err.read(&mut buf_err) => {
                    let n = res.unwrap();
                    &buf_err[..n]
                },
            };

            let mut ui = ui.lock().await;
            ui.push_output(buf);
            ui.draw().await.unwrap();
        }
    };

    tokio::select!(
        res = stdin2proc => {
            println!("res : {:?}", res);
        },
        res = proc2stdout => {
            println!("res : {:?}", res);
        }
    );
}

pub struct RawStdout {
    old_ios: termios::Termios,
    w: tokio::io::Stdout,
}

impl RawStdout {
    fn new(w: tokio::io::Stdout) -> nix::Result<RawStdout> {
        let old_ios = termios::tcgetattr(w.as_raw_fd())?;
        let mut new_ios = old_ios.clone();

        termios::cfmakeraw(&mut new_ios);
        termios::tcsetattr(w.as_raw_fd(), termios::SetArg::TCSAFLUSH, &new_ios)?;

        Ok(RawStdout { old_ios, w })
    }
}

impl Drop for RawStdout {
    fn drop(&mut self) {
        let res = termios::tcsetattr(
            self.w.as_raw_fd(),
            termios::SetArg::TCSADRAIN,
            &self.old_ios,
        );

        if let Err(e) = res {
            println!("an error occured un-rawing terminal: {:?}", e);
        }
    }
}

impl std::ops::Deref for RawStdout {
    type Target = tokio::io::Stdout;

    fn deref(&self) -> &Self::Target {
        &self.w
    }
}

impl std::ops::DerefMut for RawStdout {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.w
    }
}

impl AsyncWrite for RawStdout {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<io::Result<usize>> {
        AsyncWrite::poll_write(std::pin::Pin::new(&mut self.w), cx, buf)
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        AsyncWrite::poll_flush(std::pin::Pin::new(&mut self.w), cx)
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        AsyncWrite::poll_shutdown(std::pin::Pin::new(&mut self.w), cx)
    }
}
