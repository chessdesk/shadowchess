<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/shadowchess-logotype.svg">
  <img src="assets/shadowchess-logotype-dark-ink.svg" alt="Shadowchess" height="48">
</picture>

Chess networks that play like human players of a given rating, mistakes included.
[Chessdesk](https://github.com/chessdesk/chessdesk) downloads and runs them.

Maia-style networks are much harder to beat than their rating claims because they
always play their most likely move. Shadowchess instead samples from the predicted
human move distribution, at a per-rating temperature fitted so that the net's
average move quality and blunder rate (judged by Stockfish) match real players'.

## Files

Each [release](https://github.com/chessdesk/shadowchess/releases) holds:

| File | What |
|------|------|
| `shadowchess.safetensors` | fp16 weights; the rating is an input. Metadata: `dim`, `depth`, `heads`, and `temperatures` (JSON, by rating) |
| `shadowchess-{500,1000,1500}.onnx` | One net per level, rating and temperature baked in. Input `planes` `[1, 64, 17]`, output `logits` `[1, 16384]` |

Sample a move from `softmax(logits)` over the legal moves. With the safetensors
weights, divide the logits by the level's temperature first.

## The network

A 7.0M-parameter transformer: one token per square, 8 blocks of width 256 with 8
heads, QK-norm and a learned per-head 64×64 attention bias, the rating added
through a small MLP. Its policy head scores moves as from-square query × to-square
key, with 4 promotion kinds (16,384 moves).

**Input**, from the side to move's view (the board is mirrored when Black moves):
per square, a one-hot piece code (empty, own P N B R Q K, opponent's P N B R Q K)
and 4 flags: en passant square, rook with castling rights, previous move's from
and to squares. **Rating** is fed as `(elo - 1100) / 400`. **Move index** is
`promotion × 4096 + from × 64 + to`, with squares mirrored for Black and queen
promotions in slot 0 (knight 1, bishop 2, rook 3).

Trained on 110M positions from September 2025 Lichess rated blitz, rapid and
classical games between players rated 400–1799.

## Results (v1.0.0)

| Level | Temperature | Move loss, humans / net | Blunders, humans / net |
|-------|-------------|-------------------------|------------------------|
| 500   | 1.051       | 0.067 / 0.067           | 9.7% / 9.8%            |
| 1000  | 1.003       | 0.050 / 0.050           | 6.8% / 6.4%            |
| 1500  | 0.838       | 0.041 / 0.041           | 4.3% / 4.4%            |

On 26k held-out positions from 1100–1799 players it predicts the human move 51–54%
of the time, 2–5 points more than Maia-2 (23.3M parameters) in every 100-point band.
