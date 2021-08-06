use termion::{
    event::{Event, Key},
    input::{MouseTerminal, TermReadEventsAndRaw},
    raw::IntoRawMode,
};

use std::io::{stdin, stdout, Write};

fn main() {
    let mut stdout = MouseTerminal::from(stdout().into_raw_mode().unwrap());

    let stdin = stdin();

    for (evt, bytes) in stdin.events_and_raw().map(Result::unwrap) {
        write!(stdout, "{:?}{:x?}\r\n", evt, bytes).unwrap();
        stdout.flush().unwrap();

        if let Event::Key(Key::Ctrl('c' | 'd')) = evt {
            break;
        }
    }
}
