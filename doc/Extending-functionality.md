Shuffler can not cover all user cases of working with annotations.
Therefore, it is essential for a user to be able to extend the functionality easily.


## Theory.

All functions associated with sub-commands share the same interface:
```python
def myFunction(cursor, args)
```

Here `cursor` is an SQLite3 cursor for the open database,
and `args` is the named namespace with parsed arguments
for `myFunction`. That makes adding sub-commands straightforward.

To add a new function and its associated sub-command, one first needs
to pick a file where the function would fit the best based on its functionality.
For example, all operations of filtering reside in `lib/subcommands/dbFilter.py`.
Then one needs to 1) write the function body, which implements the interface,
2) write the parser, and 3) register the parser in the `add_parsers` function.
Below is the skeleton for function `myFilter`. Replace `< ... >` with your code.

```python
# Register your new parser in function add_parsers.
def add_parsers(subparsers):
  < ... >
  myFilterParser(subparsers)

# Write the parser for command-line arguments.
def myFilterParser(subparsers):
  parser = subparsers.add_parser('myFilter',
    description='My new operation')
  parser.set_defaults(func=myFilter)
  parser.add_argument('--mandatory_arg', required=True)
  parser.add_argument('--extra_arg', type=int, default=1)
  < ... >

# Write the implementation of your function.
def myFilter (cursor, args):
  ''' The implementation of sub-command myFilter.
  Args:
    cursor:  SQLite3 cursor
    args:    Command-line arguments parsed according to myFilterParser
  Returns:
    None
  '''
  < ... >
```


## Example.

Let's say we have a property in the `properties` table, that encodes an integer angle from 0 to 359.
However, a user may enter any number from -360 to 359.
We would like to have a function that would bring a number to the right standard: from 0 to 359. The function take a angle modulo 360. We will call the subcommand `propertyModulo360`.

1. The function modifies annotations, therefore, it will be placed into the file `dbModify.py`.

2. We change the function `add_parsers` in file `dbModify.py` by adding one line to the end of the function:

```python
def add_parsers(subparsers):
  < ... >
  propertyModulo360Parser(subparsers)
```

3. The subcommand will use one argument: the name of the property to be changed.
In the end of file `dbModify.py`, we will add the following lines.

```python
def propertyModulo360Parser(subparsers):
  parser = subparsers.add_parser('propertyModulo360',
    description='Change the value of the provided property to "value mod 360."')
  parser.set_defaults(func=propertyModulo360)
  parser.add_argument('--property', required=True,
    help='The name of the property to take modulo 360 of.')
  < ... >
```

4. Finally, we will write the body of the function.
It goes in the end of `lib/subcommands/dbModify.py`, right after `propertyModulo360Parser`.

```python
def propertyModulo360 (cursor, args):
  # Print out some statistics.
  cursor.execute('SELECT COUNT(1) FROM properties WHERE key=?', (args.property,))
  logging.info('Will take modulo of %d values' % c.fetchone()[0])

  # Update the value, given that all values are recorded as text.
  cursor.execute('UPDATE properties SET value=CAST(CAST(value as INTEGER) % 360) as TEXT) WHERE key=?', (args.property,))
```

That's all. Now you can invoke this sub-command.
For example, that's how to get the help about the subcommand.

```bash
./shuffler.py propertyModulo360 -h
```
