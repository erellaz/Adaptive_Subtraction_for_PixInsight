// ----------------------------------------------------------------------------
// AdaptiveSubtraction.js  -  PixInsight JavaScript Runtime (PJSR) script
// ----------------------------------------------------------------------------
//
// Adaptive subtraction of a reference image B from a primary image A
// (monochrome, same geometry, registered).
//
// Model:     A(x,y) ~ (f * B)(x,y) + b(x,y)
//
//   f is either a scalar gain a(x,y), no kernel         (matching filter radius 0)
//   or a local (2q+1)x(2q+1) matching filter f(x,y)    (matching filter radius q >= 1)
//   b(x,y) is a local offset (affine model) or zero.
//
// Modes:
//   Adaptive subtraction   R = A - ( f*B + b )
//        Removes everything B predicts AND the local offset. Behaves as a
//        high-pass (cutoff ~ 2r) on whatever B does not explain.
//   Star subtraction       R = A - f*B
//        B is a star model on a zero background. The offset b is fitted only
//        as a nuisance parameter (it absorbs local sky and nebula so that the
//        scale / filter estimate is unbiased) and is NOT subtracted: sky and
//        nebula are preserved.
//
// Estimation:
//   Scalar (q = 0): windowed least squares with box sums (2r+1)^2, ridge-
//     regularized toward the global fit, coefficient maps smoothed with the
//     same window (guided-filter formulation).
//   Filter (q >= 1): normal equations accumulated per block (block ~ (2r+1)/5),
//     summed over 5x5 blocks (~ (2r+1)^2 window), ridge-regularized toward the
//     global filter, solved per block and bilinearly interpolated per pixel.
//     The filter absorbs PSF width/shape mismatch and sub-pixel shifts.
//   Norm: L2 (single fit), L2 with k*sigma rejection (residuals beyond
//   k*1.4826*MAD excluded from the next fit), or L1 by IRLS (weights
//   min(1, delta/|r|), delta = c*sigma: quadratic within delta, L1 beyond).
//
// Masks:
//   Pixels with A == 0 (registration borders) are excluded and output as 0.
//   In adaptive-subtraction mode, B == 0 is also treated as missing; in star
//   mode it is not (a star model is legitimately zero between stars).
//   Pixels of A >= saturation level are excluded from the fit (not the output).
//
// Memory: ~40 bytes/pixel (scalar), ~30 bytes/pixel (filter).
// ----------------------------------------------------------------------------

#feature-id    AdaptiveSubtraction : BlackRig > AdaptiveSubtraction
#feature-info  Adaptive (locally matched, robust least-squares) subtraction of a reference image or star model from a primary image, with optional local matching filter.

#include <pjsr/Sizer.jsh>
#include <pjsr/FrameStyle.jsh>
#include <pjsr/TextAlign.jsh>
#include <pjsr/NumericControl.jsh>
#include <pjsr/UndoFlag.jsh>
#include <pjsr/StdButton.jsh>
#include <pjsr/StdIcon.jsh>

var TITLE   = "Adaptive Subtraction";
var VERSION = "1.2";

// ----------------------------------------------------------------------------
// Parameters
// ----------------------------------------------------------------------------

var P = {
   primaryView   : null,   // A
   referenceView : null,   // B
   starMode      : false,  // false: A - (f*B + b) ; true: A - f*B (star subtraction)
   taps          : 1,      // matching filter half-size q (0 = scalar gain, no kernel)
   radius        : 100,    // window half-size in pixels (window ~ 2r+1)
   lambda        : 0.10,   // ridge regularization toward the global fit
   affine        : true,   // fit offset b(x,y)
   robust        : 2,      // 0 = L2, 1 = L2 with k*sigma rejection, 2 = L1 (IRLS)
   iterations    : 4,      // total fits for robust norms (L2 always uses 1)
   kSigma        : 3.0,    // rejection threshold (robust = 1)
   l1Delta       : 0.5,    // IRLS floor, in units of residual sigma (robust = 2)
   minFraction   : 0.25,   // min fraction of fit pixels in a window
   saturation    : 1.0,    // A >= saturation excluded from the fit (1 = off)
   pedestal      : 0.05,   // added to the output (adaptive-subtraction mode only)
   truncate      : true,   // clip output to [0,1]
   outScale      : false,  // also output a(x,y) (or sum of filter taps)
   outOffset     : false,  // also output b(x,y)
   outPrediction : false   // also output what was subtracted
};

// ----------------------------------------------------------------------------
// Core numerics (no PixInsight dependencies)
// ----------------------------------------------------------------------------

// valid: pixels that get an output value; fit: pixels usable for fitting.
function makeMasks( A, B, N, p )
{
   var valid = new Uint8Array( N ), fit = new Uint8Array( N );
   var bZeroMissing = !p.starMode;
   var satOn = p.saturation < 1;
   for ( var i = 0; i < N; ++i )
   {
      var ok = A[i] != 0 && (!bZeroMissing || B[i] != 0);
      valid[i] = ok ? 1 : 0;
      fit[i] = (ok && (!satOn || A[i] < p.saturation)) ? 1 : 0;
   }
   return { valid: valid, fit: fit };
}

// Number of in-bounds samples along one axis for a window of half-size r.
function axisCounts( n, r )
{
   var c = new Float32Array( n );
   for ( var i = 0; i < n; ++i )
      c[i] = Math.min( i + r, n - 1 ) - Math.max( i - r, 0 ) + 1;
   return c;
}

// Separable box SUM over a (2r+1)x(2r+1) window clipped at the borders.
// Running sums are accumulated in double precision; storage is Float32.
function boxSum( src, W, H, r, dst, tmp )
{
   var x, y, o, s, xe, ye, oa, od;

   for ( y = 0; y < H; ++y )
   {
      o = y*W;
      s = 0;
      for ( x = 0, xe = Math.min( r, W - 1 ); x <= xe; ++x )
         s += src[o + x];
      for ( x = 0; x < W; ++x )
      {
         tmp[o + x] = s;
         if ( x + r + 1 < W )
            s += src[o + x + r + 1];
         if ( x - r >= 0 )
            s -= src[o + x - r];
      }
   }

   var acc = new Float64Array( W );
   for ( y = 0, ye = Math.min( r, H - 1 ); y <= ye; ++y )
   {
      o = y*W;
      for ( x = 0; x < W; ++x )
         acc[x] += tmp[o + x];
   }
   for ( y = 0; y < H; ++y )
   {
      o = y*W;
      for ( x = 0; x < W; ++x )
         dst[o + x] = acc[x];
      if ( y + r + 1 < H )
      {
         oa = (y + r + 1)*W;
         for ( x = 0; x < W; ++x )
            acc[x] += tmp[oa + x];
      }
      if ( y - r >= 0 )
      {
         od = (y - r)*W;
         for ( x = 0; x < W; ++x )
            acc[x] -= tmp[od + x];
      }
   }
}

// Median and MAD-based sigma of r over masked pixels (subsampled to ~1e6).
function robustStats( r, mask, N )
{
   var step = Math.max( 1, Math.floor( N/1000000 ) );
   var s = [];
   for ( var i = 0; i < N; i += step )
      if ( mask[i] )
         s.push( r[i] );
   if ( s.length < 10 )
      return { median: 0, sigma: 0 };
   var cmp = function( a, b ) { return a - b; };
   s.sort( cmp );
   var med = s[s.length >> 1];
   for ( var j = 0; j < s.length; ++j )
      s[j] = Math.abs( s[j] - med );
   s.sort( cmp );
   return { median: med, sigma: 1.4826*s[s.length >> 1] };
}

// Number of fits actually performed for the chosen norm.
function numberOfFits( p )
{
   return (p.robust == 0) ? 1 : Math.max( 1, p.iterations );
}

// Robust reweighting shared by both engines. Returns false if sigma == 0.
//   robust = 1: hard rejection, w = 1 if |r - med| <= k*sigma, else 0.
//   robust = 2: L1 by iteratively reweighted least squares,
//               w = 1 for |r| <= delta, w = delta/|r| beyond, delta = l1Delta*sigma.
//               Minimizing sum w*r^2 with these weights converges to the
//               minimum of sum rho(r), rho quadratic within delta and linear
//               (L1) beyond it; delta -> 0 gives pure L1.
function reweight( resid, fit, wgt, N, p, log )
{
   var st = robustStats( resid, fit, N );
   if ( st.sigma <= 0 )
   {
      log( "  Residual sigma is zero; stopping reweighting." );
      return false;
   }
   var i, nf = 0;
   if ( p.robust == 1 )
   {
      var lim = p.kSigma*st.sigma, kept = 0;
      for ( i = 0; i < N; ++i )
      {
         if ( fit[i] )
         {
            ++nf;
            if ( Math.abs( resid[i] - st.median ) <= lim ) { wgt[i] = 1; ++kept; }
            else wgt[i] = 0;
         }
         else
            wgt[i] = 0;
      }
      log( "  Residual sigma (MAD) = " + st.sigma.toExponential( 3 ) +
           ", kept " + (100*kept/Math.max( nf, 1 )).toFixed( 1 ) + "% of fit pixels" );
   }
   else
   {
      var delta = Math.max( p.l1Delta, 1e-6 )*st.sigma, sumw = 0, full = 0;
      for ( i = 0; i < N; ++i )
      {
         if ( fit[i] )
         {
            ++nf;
            var ar = Math.abs( resid[i] ), w;
            if ( ar <= delta ) { w = 1; ++full; }
            else w = delta/ar;
            wgt[i] = w;
            sumw += w;
         }
         else
            wgt[i] = 0;
      }
      log( "  Residual sigma (MAD) = " + st.sigma.toExponential( 3 ) +
           ", L1 weights: mean " + (sumw/Math.max( nf, 1 )).toFixed( 3 ) +
           ", " + (100*full/Math.max( nf, 1 )).toFixed( 1 ) + "% within delta" );
   }
   return true;
}

// ---- Scalar engine: A ~ a*B + b ---------------------------------------------

function adaptiveFit( A, B, W, H, p, log, masks )
{
   var N = W*H, r = p.radius, i, x, y, v, nFits = numberOfFits( p );
   if ( !log )
      log = function() {};
   if ( !masks )
      masks = makeMasks( A, B, N, p );
   var valid = masks.valid, fit = masks.fit;

   var cx = axisCounts( W, r ), cy = axisCounts( H, r );

   var wgt = new Float32Array( N );
   for ( i = 0; i < N; ++i )
      wgt[i] = fit[i];

   var prod = new Float32Array( N ), tmp = new Float32Array( N );
   var Sw  = new Float32Array( N ), SA  = new Float32Array( N ), SB = new Float32Array( N );
   var SAB = new Float32Array( N ), SBB = new Float32Array( N );
   var ag = 0, bg = 0;

   for ( var it = 0; it < nFits; ++it )
   {
      var gw = 0, gA = 0, gB = 0, gAB = 0, gBB = 0;

      boxSum( wgt, W, H, r, Sw, tmp );

      for ( i = 0; i < N; ++i ) { gw += wgt[i]; v = wgt[i]*A[i]; prod[i] = v; gA += v; }
      boxSum( prod, W, H, r, SA, tmp );

      for ( i = 0; i < N; ++i ) { v = wgt[i]*B[i]; prod[i] = v; gB += v; }
      boxSum( prod, W, H, r, SB, tmp );

      for ( i = 0; i < N; ++i ) { v = wgt[i]*A[i]*B[i]; prod[i] = v; gAB += v; }
      boxSum( prod, W, H, r, SAB, tmp );

      for ( i = 0; i < N; ++i ) { v = wgt[i]*B[i]*B[i]; prod[i] = v; gBB += v; }
      boxSum( prod, W, H, r, SBB, tmp );

      if ( gw < 1 )
         throw new Error( "No valid pixels left for fitting." );

      var mA = gA/gw, mB = gB/gw, eps;
      if ( p.affine )
      {
         var vB = gBB/gw - mB*mB;
         ag  = (gAB/gw - mA*mB)/Math.max( vB, 1e-30 );
         bg  = mA - ag*mB;
         eps = p.lambda*vB;
      }
      else
      {
         ag  = gAB/Math.max( gBB, 1e-30 );
         bg  = 0;
         eps = p.lambda*gBB/gw;
      }

      // Local coefficients, in place: SAB <- a, SBB <- b
      for ( y = 0; y < H; ++y )
         for ( x = 0; x < W; ++x )
         {
            i = y*W + x;
            var n = cx[x]*cy[y], sw = Sw[i], a, b;
            if ( sw < p.minFraction*n )
            {
               a = ag;
               b = bg;
            }
            else
            {
               var la = SA[i]/sw, lb = SB[i]/sw, lab = SAB[i]/sw, lbb = SBB[i]/sw;
               if ( p.affine )
               {
                  var cov = lab - la*lb, vr = lbb - lb*lb;
                  if ( vr < 0 )
                     vr = 0;
                  a = (cov + eps*ag)/Math.max( vr + eps, 1e-30 );
                  b = la - a*lb;
               }
               else
               {
                  a = (lab + eps*ag)/Math.max( lbb + eps, 1e-30 );
                  b = 0;
               }
            }
            SAB[i] = a;
            SBB[i] = b;
         }

      // Smooth coefficient maps: SA <- a~, SB <- b~
      boxSum( SAB, W, H, r, SA, tmp );
      boxSum( SBB, W, H, r, SB, tmp );

      var aMin = Infinity, aMax = -Infinity;
      for ( y = 0; y < H; ++y )
         for ( x = 0; x < W; ++x )
         {
            i = y*W + x;
            var nn = cx[x]*cy[y];
            SA[i] /= nn;
            SB[i] /= nn;
            prod[i] = A[i] - (SA[i]*B[i] + SB[i]);
            if ( valid[i] )
            {
               if ( SA[i] < aMin ) aMin = SA[i];
               if ( SA[i] > aMax ) aMax = SA[i];
            }
         }

      log( "Iteration " + (it + 1) + "/" + nFits +
           ": global a = " + ag.toFixed( 5 ) + ", b = " + bg.toExponential( 3 ) +
           ", local a in [" + aMin.toFixed( 5 ) + ", " + aMax.toFixed( 5 ) + "]" );

      if ( it < nFits - 1 )
      {
         // wgt is Float32 here; reweight writes 0/1
         if ( !reweight( prod, fit, wgt, N, p, log ) )
            break;
      }
   }

   return { residual: prod, scale: SA, offset: SB, valid: valid, work: tmp };
}

// ---- Matching-filter engine: A ~ f*B + b ------------------------------------

// Solve n x n system in place (Gaussian elimination, partial pivoting).
function solveLinear( n, G, h )
{
   var i, j, k;
   for ( k = 0; k < n; ++k )
   {
      var p = k, mx = Math.abs( G[k*n + k] );
      for ( i = k + 1; i < n; ++i )
      {
         var av = Math.abs( G[i*n + k] );
         if ( av > mx ) { mx = av; p = i; }
      }
      if ( !(mx > 1e-300) )
         return null;
      if ( p != k )
      {
         for ( j = 0; j < n; ++j )
         {
            var t = G[k*n + j]; G[k*n + j] = G[p*n + j]; G[p*n + j] = t;
         }
         var th = h[k]; h[k] = h[p]; h[p] = th;
      }
      var d = G[k*n + k];
      for ( i = k + 1; i < n; ++i )
      {
         var f = G[i*n + k]/d;
         if ( f != 0 )
         {
            for ( j = k; j < n; ++j )
               G[i*n + j] -= f*G[k*n + j];
            h[i] -= f*h[k];
         }
      }
   }
   var xs = new Float64Array( n );
   for ( i = n - 1; i >= 0; --i )
   {
      var s = h[i];
      for ( j = i + 1; j < n; ++j )
         s -= G[i*n + j]*xs[j];
      xs[i] = s/G[i*n + i];
      if ( !isFinite( xs[i] ) )
         return null;
   }
   return xs;
}

function matchingFilterFit( A, B, W, H, p, log, masks )
{
   var N = W*H, q = p.taps, i, x, y, k, m, l, dx, dy, nFits = numberOfFits( p );
   if ( !log )
      log = function() {};
   if ( !masks )
      masks = makeMasks( A, B, N, p );
   var valid = masks.valid, fit = masks.fit;

   var D = 2*q + 1, K = D*D, M = K + (p.affine ? 1 : 0);
   var T = M*(M + 1)/2, S = 1 + M + T;
   var c = (K - 1) >> 1;                       // centre tap

   var bs  = Math.max( 4, Math.round( (2*p.radius + 1)/5 ) ), R = 2;
   var nbx = Math.ceil( W/bs ), nby = Math.ceil( H/bs ), NB = nbx*nby;

   var offs = new Int32Array( K ), odx = new Int32Array( K ), ody = new Int32Array( K );
   k = 0;
   for ( dy = -q; dy <= q; ++dy )
      for ( dx = -q; dx <= q; ++dx )
      {
         offs[k] = dy*W + dx; odx[k] = dx; ody[k] = dy; ++k;
      }

   var barea = new Float64Array( NB );
   for ( var by = 0; by < nby; ++by )
      for ( var bx = 0; bx < nbx; ++bx )
         barea[by*nbx + bx] = Math.min( bs, W - bx*bs )*Math.min( bs, H - by*bs );

   var wgt    = new Float32Array( N );
   for ( i = 0; i < N; ++i )
      wgt[i] = fit[i];
   var mom    = new Float64Array( NB*S );
   var theta  = new Float64Array( NB*M );
   var thg    = new Float64Array( M );
   var resid  = new Float32Array( N ), scale = new Float32Array( N ), offset = new Float32Array( N );
   var z      = new Float64Array( M );
   var acc    = new Float64Array( S );
   var Gm     = new Float64Array( M*M ), hm = new Float64Array( M );
   if ( p.affine )
      z[M - 1] = 1;

   function unpack( sw )
   {
      for ( var a = 0; a < M; ++a )
         hm[a] = acc[1 + a]/sw;
      var idx = 1 + M;
      for ( var a2 = 0; a2 < M; ++a2 )
         for ( var b2 = a2; b2 < M; ++b2 )
         {
            var v = acc[idx++]/sw;
            Gm[a2*M + b2] = v;
            Gm[b2*M + a2] = v;
         }
   }

   // Per-column bilinear interpolation weights (block centres at (b+0.5)*bs-0.5)
   var bx0s = new Int32Array( W ), bx1s = new Int32Array( W ), txs = new Float64Array( W );
   for ( x = 0; x < W; ++x )
   {
      var fx = (x + 0.5)/bs - 0.5, b0 = Math.floor( fx ), t = fx - b0;
      if ( b0 < 0 ) { b0 = 0; t = 0; }
      if ( b0 >= nbx - 1 ) { b0 = nbx - 1; t = 0; }
      bx0s[x] = b0; bx1s[x] = Math.min( b0 + 1, nbx - 1 ); txs[x] = t;
   }
   var rowCoef = new Float64Array( nbx*M );

   for ( var it = 0; it < nFits; ++it )
   {
      // ---- Accumulate per-block normal equations
      for ( i = 0; i < mom.length; ++i )
         mom[i] = 0;
      for ( y = q; y < H - q; ++y )
      {
         var brow = Math.floor( y/bs )*nbx;
         for ( x = q; x < W - q; ++x )
         {
            i = y*W + x;
            var wi = wgt[i];
            if ( wi == 0 )
               continue;
            var base = (brow + Math.floor( x/bs ))*S;
            for ( k = 0; k < K; ++k )
               z[k] = B[i + offs[k]];
            var wa = wi*A[i];
            mom[base] += wi;
            for ( m = 0; m < M; ++m )
               mom[base + 1 + m] += wa*z[m];
            var idx = base + 1 + M;
            for ( m = 0; m < M; ++m )
            {
               var zm = wi*z[m];
               for ( l = m; l < M; ++l )
                  mom[idx++] += zm*z[l];
            }
         }
      }

      // ---- Global solution (fallback and ridge target)
      for ( m = 0; m < S; ++m )
         acc[m] = 0;
      for ( var nb = 0; nb < NB; ++nb )
         for ( m = 0; m < S; ++m )
            acc[m] += mom[nb*S + m];
      if ( acc[0] < 4*M )
         throw new Error( "Not enough valid pixels for a " + D + "x" + D + " matching filter." );
      unpack( acc[0] );
      var varB = p.affine ? Gm[c*M + c] - Gm[c*M + M - 1]*Gm[c*M + M - 1] : Gm[c*M + c];
      if ( !(varB > 0) )
         varB = 1e-30;
      var eps = p.lambda*varB, tiny = 1e-9*varB;
      for ( k = 0; k < K; ++k )
         Gm[k*M + k] += tiny;
      var sol = solveLinear( M, Gm, hm );
      if ( sol == null )
         throw new Error( "Global matching-filter system is singular (is B empty?)." );
      for ( m = 0; m < M; ++m )
         thg[m] = sol[m];

      // ---- Local solutions per block over (2R+1)^2 neighbouring blocks
      var nFallback = 0;
      for ( var cby = 0; cby < nby; ++cby )
         for ( var cbx = 0; cbx < nbx; ++cbx )
         {
            for ( m = 0; m < S; ++m )
               acc[m] = 0;
            var area = 0;
            for ( var yy = Math.max( 0, cby - R ); yy <= Math.min( nby - 1, cby + R ); ++yy )
               for ( var xx = Math.max( 0, cbx - R ); xx <= Math.min( nbx - 1, cbx + R ); ++xx )
               {
                  var nbi = yy*nbx + xx, o2 = nbi*S;
                  area += barea[nbi];
                  for ( m = 0; m < S; ++m )
                     acc[m] += mom[o2 + m];
               }
            var tb = (cby*nbx + cbx)*M, loc = null;
            if ( acc[0] >= p.minFraction*area && acc[0] >= 4*M )
            {
               unpack( acc[0] );
               for ( k = 0; k < K; ++k )
               {
                  Gm[k*M + k] += eps + tiny;
                  hm[k] += eps*thg[k];
               }
               loc = solveLinear( M, Gm, hm );
            }
            if ( loc == null )
            {
               ++nFallback;
               for ( m = 0; m < M; ++m )
                  theta[tb + m] = thg[m];
            }
            else
               for ( m = 0; m < M; ++m )
                  theta[tb + m] = loc[m];
         }

      // ---- Prediction with bilinearly interpolated coefficients
      var sMin = Infinity, sMax = -Infinity;
      for ( y = 0; y < H; ++y )
      {
         var fy = (y + 0.5)/bs - 0.5, by0 = Math.floor( fy ), ty = fy - by0;
         if ( by0 < 0 ) { by0 = 0; ty = 0; }
         if ( by0 >= nby - 1 ) { by0 = nby - 1; ty = 0; }
         var by1 = Math.min( by0 + 1, nby - 1 );
         for ( var bxx = 0; bxx < nbx; ++bxx )
         {
            var r0 = (by0*nbx + bxx)*M, r1 = (by1*nbx + bxx)*M, rc = bxx*M;
            for ( m = 0; m < M; ++m )
               rowCoef[rc + m] = (1 - ty)*theta[r0 + m] + ty*theta[r1 + m];
         }
         var yInner = y >= q && y < H - q;
         for ( x = 0; x < W; ++x )
         {
            i = y*W + x;
            var c0 = bx0s[x]*M, c1 = bx1s[x]*M, tx = txs[x], ux = 1 - tx;
            var inner = yInner && x >= q && x < W - q;
            var pred = 0, sumTaps = 0;
            for ( k = 0; k < K; ++k )
            {
               var ck = ux*rowCoef[c0 + k] + tx*rowCoef[c1 + k];
               sumTaps += ck;
               var bv;
               if ( inner )
                  bv = B[i + offs[k]];
               else
               {
                  var xk = Math.min( W - 1, Math.max( 0, x + odx[k] ) );
                  var yk = Math.min( H - 1, Math.max( 0, y + ody[k] ) );
                  bv = B[yk*W + xk];
               }
               pred += ck*bv;
            }
            var off = p.affine ? ux*rowCoef[c0 + M - 1] + tx*rowCoef[c1 + M - 1] : 0;
            resid[i]  = A[i] - pred - off;
            scale[i]  = sumTaps;
            offset[i] = off;
            if ( valid[i] )
            {
               if ( sumTaps < sMin ) sMin = sumTaps;
               if ( sumTaps > sMax ) sMax = sumTaps;
            }
         }
      }

      // ---- Diagnostics: global filter, DC gain, centroid shift
      var dc = 0, sx = 0, sy = 0;
      for ( k = 0; k < K; ++k )
      {
         dc += thg[k]; sx += odx[k]*thg[k]; sy += ody[k]*thg[k];
      }
      log( "Iteration " + (it + 1) + "/" + nFits + ": global " + D + "x" + D +
           " filter, DC gain = " + dc.toFixed( 5 ) +
           ", shift of A vs B = (" + (-sx/dc).toFixed( 3 ) + ", " + (-sy/dc).toFixed( 3 ) + ") px" +
           (p.affine ? ", b = " + thg[M - 1].toExponential( 3 ) : "") +
           ", local DC gain in [" + sMin.toFixed( 5 ) + ", " + sMax.toFixed( 5 ) + "]" +
           (nFallback > 0 ? ", " + nFallback + "/" + NB + " blocks used global filter" : "") );
      for ( dy = 0; dy < D; ++dy )
      {
         var row = "    ";
         for ( dx = 0; dx < D; ++dx )
         {
            var tv = thg[dy*D + dx];
            row += (tv >= 0 ? " " : "") + tv.toFixed( 4 ) + "  ";
         }
         log( row );
      }

      if ( it < nFits - 1 )
         if ( !reweight( resid, fit, wgt, N, p, log ) )
            break;
   }

   return { residual: resid, scale: scale, offset: offset, valid: valid, work: null };
}

// ----------------------------------------------------------------------------
// PixInsight I/O
// ----------------------------------------------------------------------------

function readChannel( view )
{
   var img = view.image;
   var W = img.width, H = img.height;
   var buf = new Float32Array( W*H );
   img.getSamples( buf, new Rect( W, H ), 0 );
   return buf;
}

function makeOutput( id, W, H, data, srcView )
{
   var win = new ImageWindow( W, H, 1, 32, true, false, id );
   var v = win.mainView;
   v.beginProcess( UndoFlag_NoSwapFile );
   v.image.setSamples( data, new Rect( W, H ), 0 );
   v.endProcess();
   try { win.keywords = srcView.window.keywords; } catch ( e ) {}
   win.show();
   return win;
}

function cleanId( s )
{
   return s.replace( /[^A-Za-z0-9_]/g, "_" );
}

function run()
{
   var vA = P.primaryView, vB = P.referenceView;
   if ( vA == null || vA.isNull || vB == null || vB.isNull )
      throw new Error( "Select both a primary and a reference image." );
   if ( vA.fullId == vB.fullId )
      throw new Error( "Primary and reference must be different views." );

   var iA = vA.image, iB = vB.image;
   if ( iA.numberOfChannels != 1 || iB.numberOfChannels != 1 )
      throw new Error( "Both images must be monochrome." );
   if ( iA.width != iB.width || iA.height != iB.height )
      throw new Error( "Images must have identical dimensions (register them first)." );

   var W = iA.width, H = iA.height, N = W*H;
   var t0 = Date.now();
   var D = 2*P.taps + 1;

   console.show();
   console.writeln( "<end><cbr><br><b>" + TITLE + " " + VERSION + "</b>" );
   console.writeln( "A = " + vA.fullId + ", B = " + vB.fullId + ", " + W + "x" + H +
                    ", mode = " + (P.starMode ? "star subtraction (A - f*B)" : "adaptive subtraction (A - (f*B + b))") +
                    ", filter = " + (P.taps == 0 ? "scalar" : D + "x" + D) +
                    ", r = " + P.radius + ", lambda = " + P.lambda +
                    ", offset = " + (P.affine ? "fitted" : "none") +
                    ", norm = " + ["L2", "L2 + " + P.kSigma + "-sigma rejection",
                                   "L1 (IRLS, delta = " + P.l1Delta + " sigma)"][P.robust] +
                    ", fits = " + numberOfFits( P ) +
                    (P.saturation < 1 ? ", saturation = " + P.saturation : "") );
   processEvents();

   var A = readChannel( vA );
   var B = readChannel( vB );
   var masks = makeMasks( A, B, N, P );
   var logf = function( s ) { console.writeln( s ); processEvents(); };

   var res = (P.taps == 0) ? adaptiveFit( A, B, W, H, P, logf, masks )
                           : matchingFilterFit( A, B, W, H, P, logf, masks );

   // What was subtracted: f*B + b (adaptive) or f*B (star mode)
   var pred = null;
   if ( P.outPrediction )
   {
      pred = new Float32Array( N );
      for ( var j = 0; j < N; ++j )
         pred[j] = res.valid[j] ? A[j] - res.residual[j] - (P.starMode ? res.offset[j] : 0) : 0;
   }
   A = null; B = null;

   var out = res.work ? res.work : new Float32Array( N );
   var lo = 0, hi = 0;
   for ( var i = 0; i < N; ++i )
   {
      if ( !res.valid[i] ) { out[i] = 0; continue; }
      var v = P.starMode ? res.residual[i] + res.offset[i]
                         : res.residual[i] + P.pedestal;
      if ( P.truncate )
      {
         if ( v < 0 ) { v = 0; ++lo; }
         else if ( v > 1 ) { v = 1; ++hi; }
      }
      out[i] = v;
   }

   var base = cleanId( vA.id );
   makeOutput( base + (P.starMode ? "_starless" : "_adaptsub"), W, H, out, vA );
   if ( P.outScale )  makeOutput( base + "_scale",  W, H, res.scale,  vA );
   if ( P.outOffset ) makeOutput( base + "_offset", W, H, res.offset, vA );
   if ( pred != null ) makeOutput( base + (P.starMode ? "_stars" : "_prediction"), W, H, pred, vA );

   if ( P.truncate )
      console.writeln( "Truncated: " + lo + " pixels below 0 (" + (100*lo/N).toFixed( 3 ) +
                       "%), " + hi + " above 1." +
                       (P.starMode ? " Pixels below 0 in star mode indicate over-subtraction."
                                   : " Raise the pedestal if the low count is not ~0.") );
   console.writeln( "Done in " + ((Date.now() - t0)/1000).toFixed( 1 ) + " s." );
}

// ----------------------------------------------------------------------------
// Dialog
// ----------------------------------------------------------------------------

function ASDialog()
{
   this.__base__ = Dialog;
   this.__base__();
   var self = this;

   var labelWidth = this.font.width( "Matching filter radius:" ) + 24;

   function rowLabel( parent, text )
   {
      var l = new Label( parent );
      l.text = text;
      l.minWidth = labelWidth;
      l.textAlignment = TextAlign_Right | TextAlign_VertCenter;
      return l;
   }

   function numeric( parent, text, min, max, precision, value, tip, setter )
   {
      var c = new NumericControl( parent );
      c.real = true;
      c.label.text = text;
      c.label.minWidth = labelWidth;
      c.setRange( min, max );
      c.slider.setRange( 0, 1000 );
      c.slider.minWidth = 250;
      c.setPrecision( precision );
      c.setValue( value );
      c.toolTip = tip;
      c.onValueUpdated = function( v ) { setter( v ); };
      return c;
   }

   function spinRow( parent, text, min, max, value, tip, setter )
   {
      var s = new SpinBox( parent );
      s.setRange( min, max );
      s.value = value;
      s.toolTip = tip;
      s.onValueUpdated = function( v ) { setter( v ); };
      var sz = new HorizontalSizer;
      sz.spacing = 4;
      sz.add( rowLabel( parent, text ) );
      sz.add( s );
      sz.addStretch();
      return sz;
   }

   function check( parent, text, value, tip, setter )
   {
      var c = new CheckBox( parent );
      c.text = text;
      c.checked = value;
      c.toolTip = tip;
      c.onCheck = function( v ) { setter( v ); };
      return c;
   }

   // --- Help
   this.helpLabel = new Label( this );
   this.helpLabel.frameStyle = FrameStyle_Box;
   this.helpLabel.margin = 6;
   this.helpLabel.wordWrapping = true;
   this.helpLabel.useRichText = true;
   this.helpLabel.text =
      "<b>" + TITLE + " v" + VERSION + "</b> &mdash; fits A &asymp; f(x,y)&lowast;B + b(x,y) " +
      "by windowed, robust (L1 or L2) least squares. f is a local matching filter (convolution " +
      "kernel) or, at radius 0, a scalar gain. " +
      "<b>Adaptive subtraction</b> outputs A &minus; (f&lowast;B + b) and high-passes whatever B " +
      "does not explain. <b>Star subtraction</b> takes B as a star model on a zero background and " +
      "outputs A &minus; f&lowast;B, keeping sky and nebula. Work on linear data.";

   // --- Views
   var active = ImageWindow.activeWindow;

   this.primaryList = new ViewList( this );
   this.primaryList.getAll();
   if ( !active.isNull )
   {
      this.primaryList.currentView = active.currentView;
      P.primaryView = active.currentView;
   }
   this.primaryList.onViewSelected = function( view ) { P.primaryView = view; };

   this.referenceList = new ViewList( this );
   this.referenceList.getAll();
   this.referenceList.onViewSelected = function( view ) { P.referenceView = view; };

   this.primarySizer = new HorizontalSizer;
   this.primarySizer.spacing = 4;
   this.primarySizer.add( rowLabel( this, "Primary (A):" ) );
   this.primarySizer.add( this.primaryList, 100 );

   this.referenceSizer = new HorizontalSizer;
   this.referenceSizer.spacing = 4;
   this.referenceSizer.add( rowLabel( this, "Reference / star model (B):" ) );
   this.referenceSizer.add( this.referenceList, 100 );

   // --- Mode
   this.modeCombo = new ComboBox( this );
   this.modeCombo.addItem( "Adaptive subtraction:  A - (f*B + b)" );
   this.modeCombo.addItem( "Star subtraction:  A - f*B  (sky and nebula kept)" );
   this.modeCombo.currentItem = P.starMode ? 1 : 0;
   this.modeCombo.toolTip =
      "<p><b>Adaptive subtraction</b>: removes the prediction and the local offset.</p>" +
      "<p><b>Star subtraction</b>: B is a star model on a zero background (same PSF and " +
      "astrometry as A). The offset is fitted as a nuisance term so the star scale is " +
      "unbiased by the sky under the stars, then kept in the output. Zeros in B are " +
      "treated as valid data in this mode.</p>";
   this.modeCombo.onItemSelected = function( index )
   {
      P.starMode = index == 1;
      self.pedestalControl.enabled = !P.starMode;
   };

   this.modeSizer = new HorizontalSizer;
   this.modeSizer.spacing = 4;
   this.modeSizer.add( rowLabel( this, "Mode:" ) );
   this.modeSizer.add( this.modeCombo, 100 );

   // --- Fit parameters
   this.tapsSizer = spinRow( this, "Matching filter radius:", 0, 3, P.taps,
      "<p>0 = scalar gain a(x,y), no kernel. 1 = 3&times;3 filter (default), 2 = 5&times;5, 3 = 7&times;7.</p>" +
      "<p>A matching filter absorbs PSF width/shape mismatch and sub-pixel shifts between " +
      "A and B (ring and dipole residuals). Cost grows with the square of the tap count; " +
      "3&times;3 is usually enough.</p>",
      function( v ) { P.taps = v; } );

   this.radiusSizer = spinRow( this, "Window radius (px):", 2, 4000, P.radius,
      "<p>Half-size r of the fitting window (~(2r+1)&sup2;). Larger = smoother, more global " +
      "matching and less leakage of A-only structure. In star mode, make it large enough to " +
      "contain dozens of stars.</p>",
      function( v ) { P.radius = v; } );

   // --- Norm / robustness
   this.normCombo = new ComboBox( this );
   this.normCombo.addItem( "L2 (plain least squares, one fit)" );
   this.normCombo.addItem( "L2 with k-sigma rejection" );
   this.normCombo.addItem( "L1 (IRLS)" );
   this.normCombo.currentItem = P.robust;
   this.normCombo.toolTip =
      "<p><b>L2</b>: one least-squares fit.</p>" +
      "<p><b>L2 with k-sigma rejection</b>: after each fit, residuals beyond " +
      "k&middot;1.4826&middot;MAD get weight 0 in the next fit.</p>" +
      "<p><b>L1 (IRLS)</b>: iteratively reweighted least squares with weights " +
      "min(1, &delta;/|r|), &delta; = c&middot;sigma. Converges to a fit that is quadratic " +
      "within &delta; and L1 beyond, so strong structure in A that B does not predict " +
      "(nebula, satellite trails, hot pixels, mismatched bright stars) pulls the filter " +
      "only linearly instead of quadratically. Smaller c = closer to pure L1.</p>";
   this.normCombo.onItemSelected = function( index )
   {
      P.robust = index;
      self.updateNormControls();
   };

   this.normSizer = new HorizontalSizer;
   this.normSizer.spacing = 4;
   this.normSizer.add( rowLabel( this, "Norm:" ) );
   this.normSizer.add( this.normCombo, 100 );

   this.iterSpin = new SpinBox( this );
   this.iterSpin.setRange( 1, 20 );
   this.iterSpin.value = P.iterations;
   this.iterSpin.toolTip =
      "<p>Total number of fits for the robust norms (the first fit is plain L2). " +
      "L1 typically needs 4&ndash;6; k-sigma rejection 2&ndash;3.</p>";
   this.iterSpin.onValueUpdated = function( v ) { P.iterations = v; };
   this.iterSizer = new HorizontalSizer;
   this.iterSizer.spacing = 4;
   this.iterSizer.add( rowLabel( this, "Iterations:" ) );
   this.iterSizer.add( this.iterSpin );
   this.iterSizer.addStretch();

   this.lambdaControl = numeric( this, "Regularization (lambda):", 0, 10, 3, P.lambda,
      "<p>Ridge weight pulling the local scale/filter toward the global solution, relative " +
      "to the global variance of B. 0 = fully local; large = global.</p>",
      function( v ) { P.lambda = v; } );

   this.kControl = numeric( this, "Rejection k (sigma):", 1, 10, 2, P.kSigma,
      "<p>Residuals beyond k&middot;1.4826&middot;MAD are excluded from the next fit.</p>",
      function( v ) { P.kSigma = v; } );

   this.l1Control = numeric( this, "L1 threshold c (sigma):", 0.05, 5, 2, P.l1Delta,
      "<p>&delta; = c&middot;1.4826&middot;MAD of the residual. Residuals below &delta; get " +
      "full weight (quadratic), larger ones weight &delta;/|r| (L1). 0.5 is a stable default; " +
      "smaller values approach pure L1 but converge more slowly.</p>",
      function( v ) { P.l1Delta = v; } );

   this.updateNormControls = function()
   {
      self.iterSpin.enabled  = P.robust != 0;
      self.kControl.enabled  = P.robust == 1;
      self.l1Control.enabled = P.robust == 2;
   };
   this.updateNormControls();

   this.satControl = numeric( this, "Saturation level:", 0, 1, 4, P.saturation,
      "<p>Pixels of A at or above this level are excluded from the fit (they are still " +
      "processed in the output). 1 = off.</p>",
      function( v ) { P.saturation = v; } );

   this.pedestalControl = numeric( this, "Output pedestal:", 0, 0.5, 4, P.pedestal,
      "<p>Constant added to the residual so negative noise is not clipped " +
      "(adaptive-subtraction mode only).</p>",
      function( v ) { P.pedestal = v; } );
   this.pedestalControl.enabled = !P.starMode;

   this.affineCheck = check( this, "Fit offset b(x,y) (affine model)", P.affine,
      "<p>Checked: A &asymp; f&lowast;B + b. Unchecked: A &asymp; f&lowast;B.</p>" +
      "<p>Keep it checked in star mode: without the offset, the sky under the stars biases " +
      "the scale high and the stars are over-subtracted.</p>",
      function( v ) { P.affine = v; } );
   this.truncateCheck = check( this, "Truncate output to [0,1]", P.truncate,
      "<p>Clip the output; the number of clipped pixels is reported on the console.</p>",
      function( v ) { P.truncate = v; } );
   this.scaleCheck = check( this, "Output scale map (a, or sum of filter taps)", P.outScale, "",
      function( v ) { P.outScale = v; } );
   this.offsetCheck = check( this, "Output offset map b(x,y)", P.outOffset, "",
      function( v ) { P.outOffset = v; } );
   this.predCheck = check( this, "Output subtracted component", P.outPrediction,
      "<p>f&lowast;B + b in adaptive mode; f&lowast;B (the fitted stars) in star mode.</p>",
      function( v ) { P.outPrediction = v; } );

   this.checksSizer = new VerticalSizer;
   this.checksSizer.spacing = 4;
   this.checksSizer.add( this.affineCheck );
   this.checksSizer.add( this.truncateCheck );
   this.checksSizer.add( this.scaleCheck );
   this.checksSizer.add( this.offsetCheck );
   this.checksSizer.add( this.predCheck );

   this.checksRow = new HorizontalSizer;
   this.checksRow.addSpacing( labelWidth + 4 );
   this.checksRow.add( this.checksSizer );
   this.checksRow.addStretch();

   // --- Buttons
   this.okButton = new PushButton( this );
   this.okButton.text = "OK";
   this.okButton.onClick = function() { this.dialog.ok(); };

   this.cancelButton = new PushButton( this );
   this.cancelButton.text = "Cancel";
   this.cancelButton.onClick = function() { this.dialog.cancel(); };

   this.buttonsSizer = new HorizontalSizer;
   this.buttonsSizer.spacing = 6;
   this.buttonsSizer.addStretch();
   this.buttonsSizer.add( this.okButton );
   this.buttonsSizer.add( this.cancelButton );

   // --- Layout
   this.sizer = new VerticalSizer;
   this.sizer.margin = 8;
   this.sizer.spacing = 6;
   this.sizer.add( this.helpLabel );
   this.sizer.addSpacing( 4 );
   this.sizer.add( this.primarySizer );
   this.sizer.add( this.referenceSizer );
   this.sizer.add( this.modeSizer );
   this.sizer.addSpacing( 4 );
   this.sizer.add( this.tapsSizer );
   this.sizer.add( this.radiusSizer );
   this.sizer.add( this.lambdaControl );
   this.sizer.addSpacing( 4 );
   this.sizer.add( this.normSizer );
   this.sizer.add( this.iterSizer );
   this.sizer.add( this.kControl );
   this.sizer.add( this.l1Control );
   this.sizer.addSpacing( 4 );
   this.sizer.add( this.satControl );
   this.sizer.add( this.pedestalControl );
   this.sizer.add( this.checksRow );
   this.sizer.addSpacing( 4 );
   this.sizer.add( this.buttonsSizer );

   this.windowTitle = TITLE;
   this.adjustToContents();
}

ASDialog.prototype = new Dialog;

// ----------------------------------------------------------------------------

function main()
{
   console.hide();
   var dlg = new ASDialog();
   for ( ;; )
   {
      if ( !dlg.execute() )
         return;
      try
      {
         run();
         return;
      }
      catch ( e )
      {
         (new MessageBox( e.message, TITLE, StdIcon_Error, StdButton_Ok )).execute();
      }
   }
}

main();
