# Why ChatGPT use Ġ as Space?

Firs look at the first 255 character on [List of Unicode Symbols](https://symbl.cc/en/unicode-table/). How many printable characters are there? If you are asked to select 256 unique characters to represent each number in the range 0-255 and the character must be printable, how would you do it?

Let's pick 256 printable characters from the beginning of the table and dump those not printable.

0000-001F are not printable. Dump them.

0020 is space and space is printable but not very distinguishable to human eyes. Dump them.

0021-007E are printable. Keep them.

007F-00AO are not printable. Dump them.

00A1-00AC are printable. Keep them.

00AD is not printable. Dump it.

00AE-00FF are printable. Keep them.

00FF is 255. Now your are reaching the final number in 2^8. However, you have only got 188 printable characters. You need more characters to fill the holes of not-printable characters.

Continue to select printable characters from next position 0100 (which is Ā). There're plenty of them starting from this position. We select 68 more and stop at 0143.

Now we should have a visible character representing space. Where is it?

On the unicode table, space is 0020 (position 32. Do forget the the first position is 0). 0100 + 0020 -> 0120, which is Ġ. 

There's a saying that Ġ is specifically chosen to represent space. No. It's coincidence.