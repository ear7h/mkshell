# mkshell

One command that turns any process into an interactive shell.

## usage

```
mkshell --input-decorator ./polite.sh cat
```

This command will open a new `mkshell` shell backed by the `cat` command, and
user input will be written to the script `./greet.sh` and the script's output
gets written to `cat`


## TODO

* render caching
* check script and return nice messages
* more ways to modify the shell with scripts
    * `--init` - gets sent to the child process when the shell starts
    * `--divider` - a script or file that gets called to create the I/O
        divider (ex. color coded pwd)
