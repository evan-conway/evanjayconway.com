# evanjayconway.com

A hand-written Jekyll site. It replaced the al-folio theme, which was too hard
to customize. The guiding principle is **use Jekyll's defaults and add nothing
that isn't needed** -- no plugins, no build tooling, no framework.

## Running it

```bash
docker compose up      # http://localhost:8080, live reloads on save
docker compose down    # stop
```

Editing `_config.yml` requires a restart; Jekyll does not reload it.

## Layout of the repo

```
_config.yml              site title, post permalink, one exclude. That's all.
index.html               landing page (/)
posts.html               post index (/posts/)
404.html                 not-found page
_posts/                  one file per post
_layouts/default.html    <html> skeleton; every page passes through it
_layouts/page.html       standalone page: <h1> + content
_layouts/post.html       post: <h1> + date + description + content
_includes/head.html      everything in <head>, incl. MathJax
_includes/header.html    site header and nav
_includes/footer.html    site footer
_includes/figure.liquid  the <img> tag used by posts
assets/css/main.css      the only stylesheet
assets/img/              images, one subdirectory per post
```

## How to do things

**Add a post.** Create `_posts/YYYY-MM-DD-slug.md`:

```markdown
---
layout: post
title: The Title
date: 2026-03-07
description: One line, shown under the title.
---

Body text here.
```

It appears on `/posts/` automatically, at `/posts/YYYY/slug/`. Post images go
in `assets/img/<slug>/`.

**Add a page.** Create `whatever.html` (or `.md`) in the repo root:

```markdown
---
layout: page
title: Books
permalink: /books/
---

Content here.
```

Then add a link to it in `_includes/header.html` if it should be in the nav.

**Change the styling.** It all lives in `assets/css/main.css`. Do not add
`<style>` blocks to layouts or `_styles:` keys to front matter -- keeping every
rule in one file is what makes restyling cheap.

## Conventions

- **Semantic HTML, essentially no classes.** Style `h2`, `article`, `figure`
  directly. The few classes that exist (`site-header`, `post-date`) are hooks
  for things CSS can't otherwise select. Avoid utility classes; they mean a
  restyle has to touch every page instead of one stylesheet.
- **Comment the non-obvious.** Several things here look removable but aren't:
  the `inlineMath` line in `head.html`, the `permalink` in `_config.yml`. Each
  has a comment saying why. Leave those comments in place.
- **Prefer an obvious edit over a clever abstraction.** Nav links are written
  out by hand rather than looped over a data file, because a three-item nav is
  easier to edit as three lines of HTML.

## Post rendering: what's load-bearing

The nanogpt-improvements post is the reason the site is built this way. It uses
markdown, GFM footnotes (~80 of them), LaTeX, and figures. Three things make
that work, and all three are easy to break:

1. **`permalink` in `_config.yml`** keeps the post's public URL stable.
2. **The `inlineMath` config in `head.html`.** kramdown turns `$$...$$` into
   `\[...\]`, which MathJax reads by default, but leaves single-`$` inline math
   as plain text -- so MathJax has to be configured to look for it.
3. **`_includes/figure.liquid`** exists because the post calls
   `{% include figure.liquid path="..." %}`. It's a one-line `<img>` wrapper,
   kept only so the post's markdown doesn't have to be rewritten.

Footnotes, fenced code, and tables come from kramdown's GFM input mode, which
is already Jekyll's default -- there is no configuration for them, and there
shouldn't be.

## Deployment

Not set up on this branch yet. The live site is still served from `main`, which
builds the old al-folio site and pushes `_site` to the `gh-pages` branch. The
custom domain depends on a `CNAME` file surviving on `gh-pages`.
