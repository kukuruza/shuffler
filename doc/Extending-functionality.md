All functions share the same interface:
```python3
def myFunction(cursor, args)
```

Here `cursor` is an SQLite3 cursor for the open database, and `args` is the named namespace with parsed arguments
for `myFunction`. That makes adding sub-commands straightforward. 

To add a new function and its associated sub-command, 
one first needs to pick a file where the function would fit the best based on its functionality. 
For example, all operations of filtering reside in `dbFilter.py`. 
Then one needs to 1) write the function body, which implements the interface, 
2) write the parser, and 3) register the parser in the `add_parsers` function.
Below is the skeleton for function `myFilter`. Replace `< ... >` with your code.

```python3
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
def myFilter (c, args):
  ''' The implementation of sub-command myFilter.
  Args:
    c:    SQLite3 cursor
    args: Command-line arguments parsed according to myFilterParser
  Returns:
    None
  '''
  < ... >
```
