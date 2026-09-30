# Paper Book — Portfolio home (design source)

Open **Portfolio Home.dc.html** in a browser to see all four views, or **PortfolioScreen.dc.html** for a single screen
(set `theme` / `layout` via the component props). Both need `support.js` next to them.

## Files
- `PortfolioScreen.dc.html` — the screen itself. One component, two layouts (desktop 1440 / phone 390), two themes.
  Template markup + a logic class at the bottom (sample data, formatters, sorting). Every feature block is
  marked with an `═══` comment.
- `Portfolio Home.dc.html` — review canvas that shows 1a–1d side by side.
- `support.js` — the small runtime that renders the `.dc.html` files (templating, `sc-for`, `sc-if`, imports).

## Feature map (brief section → block in PortfolioScreen)
| Brief | Block comment |
|---|---|
| §3 rule 1 Paper-only | GLOBAL · PAPER-ONLY BAND |
| §4 Global chips | GLOBAL · TOP BAR / PHONE · HEADER |
| §5.1 Header metrics | PORTFOLIO · HERO, SECONDARY METRICS |
| §5.1 Trading & risk strip | WHAT HAPPENS NEXT |
| §5.1 Intraday chart | INTRADAY CHART |
| §5.1 Holdings table | HOLDINGS TABLE / PHONE · HOLDINGS LIST |
| §5.1 Sector views | SECTOR VIEW / PHONE · SECTOR BARS |
| §5.1 Trade log | TRADES |
| §5.1 Honesty footer | HONESTY FOOTER |

## Tokens (see the <style> block at the top)
Light and dark sets; switch with `data-pt="light|dark"` on the root.
- Text: --ink, --ink2, --ink3 · Surfaces: --bg, --surface, --sunk · Lines: --line, --line2, --track, --hover
- Profit/loss: --up (teal, hue 162), --down (vermilion, hue 32) — always paired with ▲/▼ and ±
- Status: --ok, --warn, --fail · Links: --accent

## Formatting rules (implemented in the logic class)
- ₹ with Indian grouping (en-IN): ₹1,00,000 · ₹99,448.68
- Signs: "+" and true minus "−" (U+2212); zero shows no sign
- Returns 2 dp (+0.56%), weights 1 dp (3.7%); whole ₹ in summaries, paise in prices
- Times IST, 24h: "30 Sep 09:16"

## Notes
- All numbers are **illustrative sample data** consistent with the brief — not live results.
- Fonts load from Google Fonts in this mock; self-host IBM Plex in production (brief §8: no phoning-home CDNs).
- Styles are inline by design (the runtime streams them); when porting to Streamlit CSS or React, lift the
  repeated literals into classes/components using the tokens above.
