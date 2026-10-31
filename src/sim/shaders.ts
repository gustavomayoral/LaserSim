// GLSL3 (WebGL2) shader sources for the deposit / blur / ramp pipeline.
// Kept as plain template strings (not three.js ShaderChunk) so the exact GLSL
// run on the GPU is easy to read end to end for this spike.

// Bilinear deposit, done as an instanced quad per sample (one quad spanning
// exactly the 2x2 texels the sample's bilinear splat covers) instead of a
// GL_POINTS sprite. A point sprite's on-screen coverage near the minimum
// supported size is not reliably specified: this pipeline measured anywhere
// from ~6% to ~400% of the expected energy depending on gl_PointSize (1, 2,
// 4, 8) on the same GPU, with no size giving exact, size-independent
// coverage. An ordinary quad uses standard triangle rasterization instead
// (one well-defined fill rule, not point-sprite-specific GPU/driver
// behavior), so each fragment's position within the quad is read directly
// from the rasterizer-interpolated UV rather than inferred from point size,
// and per-sample weights are guaranteed to sum to 1 (COD-251/COD-252's
// platform-dependent fluence loss).
export const depositVertex = /* glsl */ `
// Local unit-quad corner, shared (not instanced) geometry: (0,0)/(1,0)/
// (0,1)/(1,1), one set of 4 vertices reused for every instance.
in vec2 position;
// Per-instance (per-sample) attributes.
in vec2 aCenter; // sample position, mm
in float aEnergy;
uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;
uniform vec2 uDomainMm;
uniform vec2 uGridSize;
out vec2 vLocalUv;
out vec2 vFrac;
out float vEnergy;

void main() {
  vec2 pixelPos = (aCenter / uDomainMm) * uGridSize;
  vec2 i0 = floor(pixelPos - 0.5);
  vFrac = pixelPos - (i0 + 0.5);
  // The quad spans texel i0 to i0+2 (i.e. the 2 texels this sample's
  // bilinear splat covers on this axis); "position" (0 or 1 per axis) picks
  // which corner of that quad this vertex is.
  vec2 texelCorner = i0 + position * 2.0;
  vec2 domainPos = (texelCorner / uGridSize) * uDomainMm;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(domainPos, 0.0, 1.0);
  vLocalUv = position;
  vEnergy = aEnergy;
}
`

export const depositFragment = /* glsl */ `
precision highp float;
in vec2 vLocalUv;
in vec2 vFrac;
in float vEnergy;
out vec4 fragColor;

void main() {
  // vLocalUv is this fragment's rasterizer-interpolated position within the
  // instance's quad (0..1 per axis): <0.5 is the near texel on that axis,
  // >=0.5 is the far one, each weighted by how close aCenter was to it.
  float wx = vLocalUv.x < 0.5 ? (1.0 - vFrac.x) : vFrac.x;
  float wy = vLocalUv.y < 0.5 ? (1.0 - vFrac.y) : vFrac.y;
  fragColor = vec4(vEnergy * wx * wy, 0.0, 0.0, 0.0);
}
`

export const fullscreenVertex = /* glsl */ `
in vec3 position;
out vec2 vUv;
void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

// Separable Gaussian blur, one direction per draw. RADIUS is baked in at
// shader-build time (see buildBlurFragment) because WebGL2 requires a
// compile-time constant loop bound.
export function buildBlurFragment(radius: number): string {
  return /* glsl */ `
precision highp float;
in vec2 vUv;
uniform sampler2D uTex;
uniform vec2 uTexelStep; // (1/gridSize, 0) or (0, 1/gridSize)
uniform float uSigmaTexels;
uniform float uInvNorm;
out vec4 fragColor;

void main() {
  float sum = 0.0;
  for (int k = -${radius}; k <= ${radius}; k++) {
    float fk = float(k);
    vec2 sampleUv = vUv + uTexelStep * fk;
    // Zero-pad taps that fall outside the domain instead of letting the
    // render target's clamp-to-edge wrap mode re-read the border texel:
    // clamping here would replay near-edge energy into multiple output
    // texels (over-counting it, not conserving it) whenever a sample sits
    // near the domain boundary — see pipeline.ts's deposit comment. Zero
    // padding lets near-edge output fall below the normalized total
    // instead, which is the physically correct direction (energy that
    // would spread past the bed edge is lost, not reflected back in).
    bool inBounds = sampleUv.x >= 0.0 && sampleUv.x <= 1.0 && sampleUv.y >= 0.0 && sampleUv.y <= 1.0;
    if (inBounds) {
      float w = exp(-0.5 * (fk / uSigmaTexels) * (fk / uSigmaTexels));
      sum += texture(uTex, sampleUv).r * w;
    }
  }
  fragColor = vec4(sum * uInvNorm, 0.0, 0.0, 0.0);
}
`
}

// Fluence -> material ramp. rgb = placeholder wood color (tan -> brown ->
// black), a = ablation height above threshold, both resolution-independent
// of the sim grid (sampled later by the heightmap material).
export const rampFragment = /* glsl */ `
precision highp float;
in vec2 vUv;
uniform sampler2D uFluence;
uniform float uScale;
uniform float uThreshold;
out vec4 fragColor;

vec3 rampColor(float t) {
  vec3 tan = vec3(0.82, 0.65, 0.42);
  vec3 brown = vec3(0.40, 0.23, 0.10);
  vec3 black = vec3(0.03, 0.02, 0.02);
  if (t < 0.5) return mix(tan, brown, t / 0.5);
  return mix(brown, black, (t - 0.5) / 0.5);
}

void main() {
  float f = texture(uFluence, vUv).r * uScale;
  float t = clamp(f, 0.0, 1.0);
  float height = clamp((f - uThreshold) / (1.0 - uThreshold + 1e-5), 0.0, 1.0);
  fragColor = vec4(rampColor(t), height);
}
`

// Heightmap display material: vertex-displaces a plane by the ramp's alpha
// (height) channel, estimates a normal by finite-differencing the same
// texture (independent of mesh segment count), and lights it with one
// directional term plus ambient.
export const heightmapVertex = /* glsl */ `
in vec3 position;
in vec2 uv;
uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 modelMatrix;
uniform sampler2D uColorHeight;
uniform float uHeightScale;
uniform vec2 uTexelStep;
uniform vec2 uDomainMm;
out vec2 vUv;
out vec3 vWorldNormal;

void main() {
  vUv = uv;
  float h = texture(uColorHeight, uv).a;
  vec3 displaced = position + vec3(0.0, 0.0, h * uHeightScale);

  float hL = texture(uColorHeight, uv - vec2(uTexelStep.x, 0.0)).a;
  float hR = texture(uColorHeight, uv + vec2(uTexelStep.x, 0.0)).a;
  float hD = texture(uColorHeight, uv - vec2(0.0, uTexelStep.y)).a;
  float hU = texture(uColorHeight, uv + vec2(0.0, uTexelStep.y)).a;
  float stepX = uDomainMm.x * uTexelStep.x;
  float stepY = uDomainMm.y * uTexelStep.y;
  vec3 tangentX = vec3(2.0 * stepX, 0.0, (hR - hL) * uHeightScale);
  vec3 tangentY = vec3(0.0, 2.0 * stepY, (hU - hD) * uHeightScale);
  vec3 localNormal = normalize(cross(tangentX, tangentY));
  vWorldNormal = mat3(modelMatrix) * localNormal;

  gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
}
`

export const heightmapFragment = /* glsl */ `
precision highp float;
in vec2 vUv;
in vec3 vWorldNormal;
uniform sampler2D uColorHeight;
uniform vec3 uLightDirWorld;
out vec4 fragColor;

void main() {
  vec3 color = texture(uColorHeight, vUv).rgb;
  vec3 n = normalize(vWorldNormal);
  float diff = max(dot(n, normalize(uLightDirWorld)), 0.0);
  vec3 lit = color * (0.35 + 0.75 * diff);
  fragColor = vec4(lit, 1.0);
}
`
