from wgpu.utils.shadertoy import Shadertoy

shader_code = """

// https://www.shadertoy.com/view/Mss3zH

// Shows how to use the mouse input (only left button supported):
//
//      mouse.xy  = mouse position during last button down
//  abs(mouse.zw) = mouse position during last button click
// sign(mouze.z)  = button is down
// sign(mouze.w)  = button is clicked


float distanceToSegment( vec2 a, vec2 b, vec2 p )
{
   vec2 pa = p - a, ba = b - a;
   float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
   return length( pa - ba*h );
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 p = fragCoord / iResolution.x;
    vec2 cen = 0.5*iResolution.xy/iResolution.x;
    vec4 m = iMouse / iResolution.x;

   vec3 col = vec3(0.0);

   if( m.z>0.0 ) // button is down
   {
      float d = distanceToSegment( m.xy, abs(m.zw), p );
      col = mix( col, vec3(1.0,1.0,0.0), 1.0-smoothstep(.004,0.008, d) );
   }
   if( m.w>0.0 ) // button click
   {
       col = mix( col, vec3(1.0,1.0,1.0), 1.0-smoothstep(0.1,0.105, length(p-cen)) );
   }

   col = mix( col, vec3(1.0,0.0,0.0), 1.0-smoothstep(0.03,0.035, length(p-    m.xy )) );
    col = mix( col, vec3(0.0,0.0,1.0), 1.0-smoothstep(0.03,0.035, length(p-abs(m.zw))) );

   fragColor = vec4( col, 1.0 );
}

"""  # noqa
shader = Shadertoy(shader_code)

if __name__ == "__main__":
    shader.show()
