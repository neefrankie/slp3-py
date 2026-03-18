# BPE

## BPE Training

The pseudo code on page 15 is very simple:

```text
function BYTE-PAIR ENCODING (strings C, number of merges k) returns vocab V
    V← all unique characters in C # initial set of tokens is characters
    for i = 1 to k do # merge tokens k times
        t_L, t_R ← Most frequent pair of adjacent tokens in C
        t_NEW ← t_L + t_R # make new token by concatenating
        V← V + t_NEW # update the vocabulary
        Replace each occurrence of t_L, t_R in C with tNEW # and update the corpus
return V
```

The book only demonstrated how the vocabulary is formed. It missed the most important part: the merge rule. Actually, the vocabulary is not that important. It is derived from the merge rule. As long as you have the merge rule, you can rebuild the vocabulary anytime.

Here is a more detailed description. Note we dropped the space char since it does not affect our understanding of the process.

Fist, count the frequency of each word: `set new new renew reset renew`, you got:

```text
set:   1
new:   2
renew: 2
reset: 1
```

Then split each word into characters:

```text
s e t: 1
n e w: 2
r e n e w: 2
r e s e t: 1
```

The you got the frequency of each ajacent pair:

```text
(s e), (e t):               1
(n e), (e w):               2
(r e), (e n), (n e), (e w): 2
(r e), (e s), (s e), (e t): 1
```

Frequency of each unique pair:

```text
(s e): 2
(e t): 2
(n e): 4
(e w): 4
(r e): 3
(e n): 2
(e s): 1
```

As human, you could see from a simple glance that `(n e)` is the most frequent pair. So merge all `(n e)` to `ne` in your character list. And then write down the `(n e)`, which is your first merging rule.

Now your corpus becomes:

```text
s e t:     1
ne w:      2
r e ne w:  2
r e s e t: 1
```

Now frequency of each ajacent pair:

```text
(s  e): 2
(e  t): 2
(ne w): 4
(r  e): 3
(e  ne): 2
(e  s): 1
```

Now merge `ne w`:

```text
s e t:     1
new:       2
r e new:   2
r e s e t: 1
```

Now your merge rule becomes:

```text
(n  e)
(ne w)
```

New frequency table:

```text
(s  e):   2
(e  t):   2
(r  e):   3
(e  new): 2
(e  s):   1
```

Note the `new: 2` is kicked out of the frequency table since it becomes a whole word and cannot generate a pair.

So we merge `r e`:

```text
s e t:     1
new:       2
re new:   2
re s e t: 1
```

Update merge rule:

```text
(n  e)
(ne w)
(r e)
```

New frequency table:

```text
(s  e):   2
(e  t):   2
(re new): 2
(re  s):  1
```

Merge `s e` now:

```text
se t:    1
new:     2
re new:  2
re se t: 1
```

Add merge rule:

```text
(n  e)
(ne w)
(r  e)
(s  e)
```

And frequency table becomes:

```text
(se t):    2
(re  new): 2
(re  se):  1
```

Merge `se t`:

```text
set:    1
new:    2
re new:  2
re set: 1
```

Add merge rule:

```text
(n  e)
(ne w)
(r  e)
(s  e)
(se t)
```

Frequency table:

```text
(re  new):  2
(re  set):  1
```

Merge `re new`:

```text
set:    1
new:    2
renew:  2
re set: 1
```

Add merge rule:

```text
(n  e)
(ne w)
(r  e)
(s  e)
(se t)
(re new)
```

Frequency table has only one pair left: `re set`. Merge it and update merge rule:

```text
(n  e)
(ne w)
(r  e)
(s  e)
(se t)
(re new)
(re set)
```

How should you use use the merge rules?

When you are given a document, split they into words, and then apply the merge rule of each character pair of each word. 

Suppose we are tokenzing the word `renew`. The characters of the word are `r e n e w`. Which pair has the highest precedence according to the merge rule? `n e` of course. So merge them: `r e ne w`.

Then which pair has the highest precedence according to the merge rule? `ne w` and merge them: `r e new`. Next comes `r e` and you got `re new`. And then `re new` becomes `renew`. You have successfuly tokenized a word based on the merge rule.

But why should we goes such a great trouble? Why not split the document by space and you also got `renew`.

Suppose you are tokenizing the word `lowest`, chances are you will got `low est` as final result rather than `lowest` since `est` usually appears more frequent than `west` or `owe` or `owest`. The more frequent a pair appears, the more sticky they are.

## How Heapq works in BPE

Industrial impelementation usually uses priority queue rather than loop the token each time.

Example merge ranks:

```python
('n', 'e'), 0
('ne', 'w'), 1
('r', 'e'), 2
('s', 'e'), 3
('se', 't'), 4
('re', 'new'), 5
('re', 'set'), 6
```

Example word to merge: `renew`

Split into characters: `renew` -> `r e n e w`

Initial pairs: `r e n e w` -> `(r, e) (e, n) (n, e) (e, w)`

Priorities:

```python
('r', 'e'), 2
('e', 'n')
('n', 'e'), 0
('e', 'w')
```

Valid set: `(0, 1, 2, 3)`.

`n e` has priority 0, so it's the first merge. We got `r e ne w`.


```python
('r', 'e'), 2
('e', 'ne')
('ne', 'w'), 1
```

Since `e ne` and `ne w` makes new pairs, we should invalidate position 1 and 2, remove the old priorities before position 2 and after it. Then add new priorities for pair 1 and pair 2, and add new valid position 1 and 2.

Now `ne w` has priority 1, so it's the next merge. We got `r e new`

```python
('r', 'e'), 2
('e', 'new')
```

Now merge `r e`. We got `re new`.

```python
('re', 'new'), 5
```

Now merge `re new`. We got `renew`.

Heap empty.