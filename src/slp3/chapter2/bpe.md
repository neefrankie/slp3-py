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