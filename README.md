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
| `shadowchess-{500,1000,1500}.onnx` | One net for each of these levels, rating and temperature baked in. Input `planes` `[1, 64, 17]`, output `logits` `[1, 16384]` |

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

## Results (v1.1.0)

Levels from 500 to 1800, every 100. Each level's temperature was fitted on about 2,000
held-out positions of players within 100 points of it, so that the net's average move loss
(judged by Stockfish at depth 10) equals theirs. One level's fit alone is noisy (about
±0.07), so the shipped temperatures are read off a quadratic through all 14 fits.

| Level | Temperature | Human blunder rate | Beats the level below by |
|-------|-------------|--------------------|--------------------------|
| 500   | 1.072       | 9.9%               |                          |
| 600   | 1.051       | 9.9%               | 134 Elo                  |
| 700   | 1.029       | 8.8%               | 125 Elo                  |
| 800   | 1.005       | 7.8%               | 118 Elo                  |
| 900   | 0.981       | 6.4%               | 73 Elo                   |
| 1000  | 0.956       | 6.9%               | 77 Elo                   |
| 1100  | 0.930       | 5.4%               | 47 Elo                   |
| 1200  | 0.903       | 5.4%               | 54 Elo                   |
| 1300  | 0.875       | 4.8%               | 47 Elo                   |
| 1400  | 0.845       | 4.1%               | 41 Elo                   |
| 1500  | 0.815       | 4.8%               | 42 Elo                   |
| 1600  | 0.784       | 4.2%               | 26 Elo                   |
| 1700  | 0.752       | 3.0%               | 35 Elo                   |
| 1800  | 0.719       | 3.4%               | 50 Elo                   |

A blunder loses at least 0.2 of the expected score. The matches are 1,000 net-vs-net games
each, so the upper levels are likely somewhat weak for their labels. Below 500 players lose
about as much per move as at 500, so there's no lower level.

On 26k held-out positions from 1100–1799 players it predicts the human move 51–54%
of the time, 2–5 points more than Maia-2 (23.3M parameters) in every 100-point band.
