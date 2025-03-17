# From Hello, Linux and Hello, Chameleon tutorial
## Log in over SSH from Local Terminal

To log in to the VM over SSH from your local terminal, follow these steps:

1. Open the terminal application installed on your computer.
2. Run the following command:
   ```sh
   ssh -i ~/.ssh/id_rsa_keyname user@{server_ip}
   ```

To validate that you are logged in to the remote host, run:

```sh
hostname
```

### Running a Command in the Remote Session

Create a file and populate it with a "hello" message:

```sh
echo "Hello from $(hostname)" > hello.txt
```

Then check the file contents:

```sh
cat hello.txt
```

---

## Transfer Files to and from Resources

While working on a remote host, we often need to transfer files between the remote host and our local filesystem.

### Using `scp` Command

The syntax for `scp` is:

```sh
scp [OPTIONS] SOURCE DESTINATION
```

#### Example:

```sh
scp -i ~/.ssh/id_rsa_keyname user@{server_ip}:/home/user/file.txt ~/Downloads/
```

### Transferring Files Through Local Terminal

1. **Transfer file from remote VM to local machine**

   ```sh
   scp -i ~/.ssh/id_rsa_keyname user@{server_ip}:/home/user/hello.txt .
   ```

   Expected output:

   ```
   hello.txt                       100%    1KB     0.1KB/s   00:00
   ```

2. **Transfer file back to remote VM**

   ```sh
   scp -i ~/.ssh/id_rsa_keyname hello.txt user@{server_ip}:/home/user/
   ```

To validate, log in to the remote host again and check the file:

```sh
cat hello.txt
```

---

## Linux Basics

### Learning the Basics of Bash Shell

#### Print a Message

```sh
echo "Hello world"
```

#### Using Variables

```sh
mymessage="hello world"
echo $mymessage
```

#### Command Substitution

```sh
myname=$(whoami)
echo "$mymessage, $myname"
echo "$mymessage, $(whoami)"
```

#### Auto-completion

Type part of a command and press `Tab` to auto-complete.

---

## History Commands

```sh
history
!1   # Re-run command 1
!!   # Re-run last command
!:0  # Command only of last command
!^   # First argument of last command
!*   # All arguments of last command
!$   # Last argument of last command
```

---

## Navigating the Filesystem

```sh
pwd      # Print working directory
ls       # List directory contents
mkdir new  # Create a new directory
cd new   # Change to new directory
cd       # Home directory
cd ..    # Move one level up
cd -     # Switch to previous directory
```

---

## Creating and Editing Files

```sh
nano newfile.txt  # Create and edit a file
cat newfile.txt   # View file contents
```

---

## Copying, Moving, and Deleting Files

```sh
cp newfile.txt copy.txt  # Copy a file
mv copy.txt mycopy.txt   # Move/Rename a file
rm mycopy.txt            # Delete a file
```

Be careful with `rm`, as deleted files cannot be recovered.

---

## Using `sudo`

`sudo` allows you to execute commands with superuser privileges. This is useful for administrative tasks like editing system files, installing software, or managing system settings.

Example:
```sh
sudo nano /etc/services
```
This command opens `/etc/services` with root access, allowing modifications.

---

## Flags, Man Pages, and Help

Many Linux commands support flags that modify their behavior. You can explore these using help options.

```sh
ls --help  # Show help for ls (brief usage information)
man ls     # Show manual for ls (detailed documentation)
```

The `man` command provides in-depth documentation for commands. Navigate using arrow keys, press `q` to exit, and use `/` to search within the manual.

---

## Retrieving Files from the Internet

You can download files from the internet using `wget`.

```sh
wget https://witestlab.poly.edu/bikes/README.txt
```

To check the contents of the downloaded file:

```sh
cat README.txt
```

If `wget` is not installed, install it using:
```sh
sudo apt install wget  # Debian-based systems
sudo yum install wget  # RHEL-based systems
```

---

## Viewing Large Files

```sh
cat /etc/services    # View entire file
head /etc/services   # View first lines
tail /etc/services   # View last lines
less /etc/services   # Scroll through file
```

To search within `less`, use `/` followed by the keyword.

```sh
grep "ftp" /etc/services  # Find lines containing 'ftp'
```

---

## I/O Redirection and Pipes

I/O redirection allows you to manipulate input and output streams in the shell.

Redirect output to a file:

```sh
echo "Hello" > output.txt  # Overwrites the file
```

Append output to an existing file:

```sh
echo "World" >> output.txt  # Adds to the file without overwriting
```

Redirect input from a file:

```sh
sort < input.txt  # Sorts the contents of input.txt
```

Using pipes (`|`) to pass output of one command as input to another:

```sh
cat /etc/services | grep "ftp"  # Filter lines containing 'ftp'
ls -l | less  # View directory listing page by page
```

Chaining multiple commands using pipes:

```sh
ip addr | grep "ether" | awk '{print $2}'  # Extract MAC addresses
```

To debug complex pipes, build them incrementally:

```sh
ip addr
ip addr | grep "ether"
ip addr | grep "ether" | awk '{print $2}'
```
