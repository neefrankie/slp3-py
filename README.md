# SLP3 Implementation

This is an attempt to implement the pseudo code described in the book [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)

## Resources

### Berkely Restaurant Project

* Paper: [Berkeley Restaurant Project](https://web.stanford.edu/~jurafsky/icslp-red.pdf)
* Dateset: [The Berkeley Restaurant Project (BeRP) Transcripts](https://github.com/wooters/berp-trans).

## Notes

- [How BPE is trained](./notes/chapter2/bpe-training.md)
- [How to use heapq](./notes/chapter2/bpe-heapq.md)
- [Why ChatGPT use Ġ as Space?](./notes/chapter2/dotg-as-space.md)

## Progress

- [Chapter 2](./src/chapter2)
    - Byte-pair training
    - Byte-pair encoding
    - Character-pair training
    - Character-pair encoding
    - Port ChatGPT's tiktoken
    - Minimum Edit Distance