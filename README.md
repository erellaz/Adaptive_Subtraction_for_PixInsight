# AdaptiveSubtraction for PixInsight

Remove an unwanted signal from an astronomical image by subtracting a **model** of it, after matching the model to the image.

Many artefacts can be modelled by a separate tool: stars rendered from a catalog, a satellite trail, diffraction spikes, a gradient, camera banding. No model is exact, though. Brightness changes along a trail, a catalog flux is not the flux through your filter, a rendered PSF is slightly too narrow, a background model misses a term. Subtract such a model directly and its errors stay in the image.

AdaptiveSubtraction takes the image **A** and the model **B** and fits, in every window, a small convolution kernel *f* and an offset *b* by robust least squares:

```
A(x,y) ≈ (f ⋆ B)(x,y) + b(x,y)        output = A − f ⋆ B
```

The kernel absorbs errors in amplitude, profile width and sub-pixel position that vary slowly across the field. Only the matched model is subtracted, so sky, nebula and everything else B does not describe stays in the image. What can be removed is limited only by what can be modelled.

## Features

- Local scalar gain, or a 3×3 to 7×7 matching filter per window
- Robust fitting: L2, L2 with k-sigma rejection, or L1 (IRLS / Huber)
- Two modes: subtract the model only (default), or the model and the local background (high-pass)
- Ridge regularization toward the global fit for windows with little model signal
- Exclusion of saturated pixels from the fit
- Diagnostic outputs: matched model, gain map, offset map; console report of the fitted gain and shift

## Installation

1. Copy `AdaptiveSubtraction.js` into a scripts folder.
2. In PixInsight: **Script › Feature Scripts… › Add**, select the folder, **Done**.
3. Run it from **Script › BlackRig › AdaptiveSubtraction**.

Inputs: A and B linear, monochrome, same dimensions and registered; B contains only the signal to remove.

## Documentation

[`AdaptiveSubtraction_how_it_works.docx`](AdaptiveSubtraction_how_it_works.docx) is the user guide and technical description. It covers the quick start, recommended settings per use case, the method with its equations, the parameters, validation and limitations.


## Author and license

BlackRig (erellaz). License: MIT.
