# Engine Working Directory

This is just a directory for cp2k to place all the files it creates (e.g. `.out`,
`.ener`, etc.). It needs to be accessible from all nodes, which is why we don't use
a directory in `/tmp`. These files are not deleted by default so the user can inspect them
if desired.