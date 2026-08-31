Eval table cell cheat sheet

```python
me.inputCell.val
str(me.inputRow)
me.exprCell.owner.path
me.inputTable.numRows
me.inputTable[me.inputRow, 'cx'] # grab another column from the same row
int(me.inputCell.val) % me.inputTable.numRows
```