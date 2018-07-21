uniform sampler2D t_video;
uniform sampler2D t_map;
uniform sampler2D t_feedback;
varying vec2 v_texcoord;
uniform float u_glitch;
uniform float u_clouds;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_mapmix;
uniform int u_mosaic;
uniform float u_feedback;
uniform float u_border;
uniform float u_master;
uniform float u_exposure;

float speed = 2.0; // Hardcoded because change regenerates noise pattern

// https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
float random (vec2 st) {
	return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

float biased_time_random (vec2 st, float bias, float time) {
	float r = fract(sin(dot((st * time).xy, vec2(12.9898, 78.233))) * 43758.5453123);
	return clamp(r + bias, 0.0, 1.0);
}

float noise (vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

#define NUM_OCTAVES 5

float fbm (vec2 st) {
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(100.0);
    // Rotate to reduce axial bias
    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.50));
    for (int i = 0; i < NUM_OCTAVES; ++i) {
        v += a * noise(st);
        st = rot * st * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

void main() {

	// Black
	vec3 black = vec3(0.0, 0.0, 0.0);

	// Normalized coordinates
	vec2 st = gl_FragCoord.xy/u_resolution.xy;

	// Clouds
	vec2 q = vec2(0.0);
	q.x = fbm(st * 3 + 0.0 * u_time * speed);
	q.y = fbm(st * 3 + vec2(1.0));
	vec2 r = vec2(0.);
	r.x = fbm(st * 3 + 1.0 * q + vec2(1.7, 9.2) + 0.15 * u_time * speed);
	r.y = fbm(st * 3 + 1.0 * q + vec2(8.3, 2.8) + 0.126 * u_time * speed);
	float f = fbm(st + r);
	vec3 color = vec3(0.0);
	color = mix(vec3(0.101961, 0.619608, 0.666667), vec3(0.666667, 0.666667, 0.498039), clamp((f * f) * 4.0, 0.0, 1.0));
	color = mix(color, vec3(0, 0, 0.164706), clamp(length(q), 0.0, 1.0));
	color = mix(color, vec3(0.666667, 1, 1), clamp(length(r.x), 0.0, 1.0));
	vec3 pre_clouds = vec3((f * f * f + 0.6 * f * f + 0.5 * f) * color);

	// https://stackoverflow.com/questions/24752734/how-to-color-a-texture-in-glsl
	float gray_clouds = dot(pre_clouds, vec3(0.299, 0.587, 0.114));
	vec3 clouds =  vec3(gray_clouds);

	// Chromatic abberation
	vec2 ca_offset = vec2(u_glitch,0.0);
	vec2 st_mod_r = st+ca_offset.xy;
	vec2 st_mod_g = st;
	vec2 st_mod_b = st+ca_offset.yx;

	// Mosaic
	for(int j = 0; j<u_mosaic;j++){
        st_mod_r *= 2.0;
        st_mod_g *= 2.0;
        st_mod_b *= 2.0;
    	st_mod_r -= 1.0;
    	st_mod_g -= 1.0;
    	st_mod_b -= 1.0;
        st_mod_r = abs(st_mod_r);
        st_mod_g = abs(st_mod_g);
        st_mod_b = abs(st_mod_b);
    }

	// Glitch and Mosaic
	vec3 video = vec3(0.0, 0.0, 0.0);
	video.r = texture2D(t_video, st_mod_r).r;
	video.g = texture2D(t_video, st_mod_g).g;
	video.b = texture2D(t_video, st_mod_b).b;

	// Map
	vec3 map = texture2D(t_map, v_texcoord).rgb;
	vec3 s0 = mix(video, map, u_mapmix);

	// Feedback
	vec3 feedback = texture2D(t_feedback, v_texcoord).rgb * u_exposure;
	vec3 s1 = mix(s0, feedback, u_feedback);

	// Master
	vec3 s2 = mix(black, s1, u_master);

	// Clouds
	vec3 s3 = mix(s2, clouds, clouds.b * u_clouds);

	// Border
	vec3 right  =  mix(s3, black, smoothstep(1.0 - u_border, 1.0, st.x));
	vec3 left =  mix(black, right, smoothstep(0.0, u_border, st.x));
	vec3 top  =  mix(left, black, smoothstep(1.0 - u_border, 1.0, st.y));
	vec3 bottom =  mix(black, top, smoothstep(0.0, u_border, st.y));

	vec3 s4 = bottom;

	gl_FragColor.rgb = s4;
	gl_FragColor.a = 1.0;

}