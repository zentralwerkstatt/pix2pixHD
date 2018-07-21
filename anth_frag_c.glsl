uniform sampler2D t_1;
uniform sampler2D t_2;
uniform sampler2D t_3;
uniform sampler2D t_4;
varying vec2 v_texcoord;
uniform vec2 u_resolution;

void main() {

	vec2 st = gl_FragCoord.xy/u_resolution.xy;

	// Divide window into four regions, scale each texture by half
	vec3 result = vec3(0.0, 0.0, 0.0);
	if (st.x > 0.5 && st.y > 0.5) {result = texture2D(t_1, (vec2(st.x - 0.5, st.y - 0.5)) * 2).rgb;}
	if (st.x < 0.5 && st.y > 0.5) {result = texture2D(t_2, (vec2(st.x, st.y - 0.5)) * 2).rgb;}
	if (st.x < 0.5 && st.y < 0.5) {result = texture2D(t_3, (vec2(st.x, st.y)) * 2).rgb;}
	if (st.x > 0.5 && st.y < 0.5) {result = texture2D(t_4, (vec2(st.x - 0.5, st.y)) * 2).rgb;}
	
	gl_FragColor.rgb = result;
	gl_FragColor.a = 1.0;
}