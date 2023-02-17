from wgpu.utils.shadertoy import Shadertoy

shader_code = """

// https://www.shadertoy.com/view/mt2XR3

// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// Created by S.Guillitte 

#define time iTime
                                 
mat2 m2 = mat2(0.8,  0.6, -0.6,  0.8);

float noise(in vec2 p){

    float res=0.;
    float f=1.;
    for( int i=0; i< 3; i++ ) 
   {      
        p=m2*p*f+.6;     
        f*=1.2;
        res+=sin(p.x+sin(2.*p.y));
   }           
   return res/3.;
}

vec3 noised(in vec2 p){//noise with derivatives
   float res=0.;
    vec2 dres=vec2(0.);
    float f=1.;
    mat2 j=m2;
   for( int i=0; i< 3; i++ ) 
   {      
        p=m2*p*f+.6;     
        f*=1.2;
        float a=p.x+sin(2.*p.y);
        res+=sin(a);
        dres+=cos(a)*vec2(1.,2.*cos(2.*p.y))*j;
        j*=m2*f;
        
   }           
   return vec3(res,dres)/3.;
}


float fbmabs( vec2 p ) {
   
   float f=.7;   
   float r = 0.0;   
    for(int i = 0;i<12;i++){   
      vec3 n = noised(p);
        r += abs(noise( p*f +n.xz)+.5)/f;       
       f *=1.45;
        p=m2*p;       
   }
   return r;
}

float sea( vec2 p ) 
{
   float f=.7;   
   float r = 0.0;   
    for(int i = 0;i<6;i++){         
        r += (1.-abs(noise( p*f -.6*time)))/f;       
       f *=1.4;
        p-=vec2(-.01,.04)*(r+.2*time/(.1-f));
   }
   return r/4.+.8;
}



float rocks(vec2 p){   
   return 1.-fbmabs(p)*.15;   
}

vec3 map( vec3 p)
{
   float d1 =p.y+ cos(.2*p.x-sin(.5*p.z))*cos(.2*p.z+sin(.3*p.x))+.5-rocks(p.xz);
    float d2 =p.y-.4*sea(p.xz);
    //dh = d2-d1;
    float d = min(d1,d2);
   return vec3(d,d1,d2);   
          
}

vec3 normalRocks(in vec2 p)
{
   const vec2 e = vec2(0.004, 0.0);
   return normalize(vec3(
      rocks(p + e.xy) - rocks(p - e.xy),
        .008,
      rocks(p + e.yx) - rocks(p - e.yx)
      ));
}

vec3 normalSea(in vec2 p)
{
   const vec2 e = vec2(0.002, 0.0);
   return normalize(vec3(
      sea(p + e.xy) - sea(p - e.xy),
        .004,
      sea(p + e.yx) - sea(p - e.yx)
      ));
}

vec3 sky(in vec2 p)
{   
    return sin(vec3(1.7,1.5,1)+ 2.-fbmabs(p*12.)*.25)+.3;
}

vec3 march(in vec3 ro, in vec3 rd)
{
   const float maxd = 35.0;
   const float precis = 0.001;
    float h = precis * 2.0;
    float t = 0.0;
   float res = -1.0;
    for(int i = 0; i < 128; i++)
    {
        if(h < precis*t || t > maxd) break;
       h = map(ro + rd * t).x;
        t += h*.5;
    }
    if(t < maxd) res = t;
    return vec3(res,map(ro + rd * t).yz);
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
   
    vec2 p = (2.0 * fragCoord.xy - iResolution.xy) / iResolution.y;
   vec3 col = vec3(0.);
      vec3 rd = normalize(vec3(p, -2.));
   vec3 ro = vec3(0.0, 2.0, -2.+.2*time);
    vec3 li = normalize(vec3(2., 2., -4.));   
    
    vec3 v = march(ro, rd);
    float t = v.x;
    float dh = v.z-v.y;
    if(t > 0.)
    {
        
        vec3 pos = ro + t * rd;       
        float k=rocks(pos.xz/2.)*2.;       
        vec3 nor = normalRocks(pos.xz/2.);
        float r = max(dot(nor, li),0.05)/2.;

        r+=.4*exp(-500.*dh*dh);
        
        col =vec3(r*k*k, r*k, r*.8);
        if(dh<0.03){
           vec3 nor = normalSea(pos.xz);
           nor = reflect(rd, nor);
            col +=vec3(0.9,.2,.05)*dh*1.;
           col += pow(max(dot(li, nor), 0.0), 5.0)*vec3(.5);
           col +=.2* sky(nor.xz/(.5+nor.y));
            
        }
       col = .1+col;
        
   }
    else //sky
        col = sky(rd.xz*(.1+rd.y));
    
      fragColor = vec4(col, 1.0);
}

"""  # noqa
shader = Shadertoy(shader_code)

if __name__ == "__main__":
    shader.show()
