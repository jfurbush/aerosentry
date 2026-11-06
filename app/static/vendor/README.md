# Vendored front-end libraries

The dashboard loads everything from this app, so it works without internet access
(for example on a demo machine with no network). Nothing is fetched from a CDN.

## plotly-basic-2.27.0.min.js

- Library: plotly.js, "basic" bundle, minified. License: MIT (see `PLOTLY-LICENSE.txt`).
- Source: the npm package `plotly.js-basic-dist-min` version 2.27.0, file `plotly-basic.min.js`.
- npm tarball sha1 (matches the checksum the npm registry publishes for this version):
  `5e91c39d8526fe8c425cde24f65d821fa8d74792`
- File sha256: `0b187f07214b18eba6015ca3b2183b76ca538e914f76a9905c3b5f3b3824c88b`
- Size: about 1 MB. The full bundle is about 3.6 MB.
- Why the basic bundle: it covers scatter, bar and pie traces. The dashboard only draws line charts. If a
  new chart type is needed, switch to `plotly.js-dist-min` of the same version.
- The dashboard used Plotly 2.27.0 from a CDN before this change, so the charts behave the same.

To check the file or re-create it:

```
npm pack plotly.js-basic-dist-min@2.27.0
tar xzf plotly.js-basic-dist-min-2.27.0.tgz
sha256sum package/plotly-basic.min.js
```

`tests/integration/test_offline.py` fails if the sha256 above stops matching the file, or if any dashboard
page refers to an external URL.

## app.css

Written for this project, not a library. It replaced the Tailwind CDN script.
