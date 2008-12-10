#include <stdio.h>
#include <math.h>

inline float floor(float x)
{
  return (float)(int)x;
}

#define noise3(x,y,z) ImprovedNoise::ins().noise(x,y,z)


class ImprovedNoise {

  int p[512];

  public:
   ImprovedNoise()
   {
    int perm[] = {151,160,137,91,90,15,
        131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
        190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
        88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
        77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
        102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
        135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
        5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
        223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
        129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
        251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
        49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
        138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180};
    for (int i=0; i < 256 ; i++)
      p[256+i] = p[i] = perm[i];
   }
   static ImprovedNoise& ins() 
   {
     static ImprovedNoise p;
     return p;
   }
   float noise(float x, float y, float z) {
      int X = (int)floor(x) & 255,                  // FIND UNIT CUBE THAT
          Y = (int)floor(y) & 255,                  // CONTAINS POINT.
          Z = (int)floor(z) & 255;
      x -= floor(x);                                // FIND RELATIVE X,Y,Z
      y -= floor(y);                                // OF POINT IN CUBE.
      z -= floor(z);
      float u = fade(x),                                // COMPUTE FADE CURVES
             v = fade(y),                                // FOR EACH OF X,Y,Z.
             w = fade(z);
      int A = p[X  ]+Y, AA = p[A]+Z, AB = p[A+1]+Z,      // HASH COORDINATES OF
          B = p[X+1]+Y, BA = p[B]+Z, BB = p[B+1]+Z;      // THE 8 CUBE CORNERS,

      return lerp(w, lerp(v, lerp(u, grad(p[AA  ], x  , y  , z   ),  // AND ADD
                                     grad(p[BA  ], x-1, y  , z   )), // BLENDED
                             lerp(u, grad(p[AB  ], x  , y-1, z   ),  // RESULTS
                                     grad(p[BB  ], x-1, y-1, z   ))),// FROM  8
                     lerp(v, lerp(u, grad(p[AA+1], x  , y  , z-1 ),  // CORNERS
                                     grad(p[BA+1], x-1, y  , z-1 )), // OF CUBE
                             lerp(u, grad(p[AB+1], x  , y-1, z-1 ),
                                     grad(p[BB+1], x-1, y-1, z-1 ))));
   }
   static float fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
   static float lerp(float t, float a, float b) { return a + t * (b - a); }
   static float grad(int hash, float x, float y, float z) {
      int h = hash & 15;                      // CONVERT LO 4 BITS OF HASH CODE
      float u = h<8 ? x : y,                 // INTO 12 GRADIENT DIRECTIONS.
             v = h<4 ? y : h==12||h==14 ? x : z;
      return ((h&1) == 0 ? u : -u) + ((h&2) == 0 ? v : -v);
   }
};


typedef unsigned char byte;
void save_tga(byte* pixels, int w, int h, const char* fname)
{
  FILE* shot;
  if((shot=fopen(fname, "wb"))==0) return ;
  byte TGAheader[12+6]={0,0,2,0,0,0,0,0,0,0,0,0,((int)(w%256)),((int)(w/256)),((int)(h%256)),((int)(h/256)),24,0}; 
  fwrite(TGAheader, sizeof(byte), 12+6, shot);
  fwrite(pixels, sizeof(byte),w*h*3, shot);
  fclose(shot);
}

struct vec3
{
  float x,y,z;
  vec3(float xx = 0.0f, float yy = 0.0f, float zz = 0.0f):
    x(xx), y(yy), z(zz) { }
  friend vec3 operator+(const vec3& a, const vec3& b) { return vec3(a.x+b.x,a.y+b.y,a.z+b.z); }
  friend vec3 operator-(const vec3& a, const vec3& b) { return vec3(a.x-b.x,a.y-b.y,a.z-b.z); }
  friend vec3 operator*(float a, const vec3& b) { return vec3(a*b.x,a*b.y,a*b.z); }
  friend float dot(const vec3& a, const vec3& b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
  friend vec3 cross(const vec3& b, const vec3& a) { return vec3((b.y * a.z) - (b.z * a.y), (b.z * a.x) - (b.x * a.z), (b.x * a.y) - (b.y * a.x)); }
  friend vec3 normalize(const vec3& a){ return (1.0f/distance(a))*a; }
  friend float distance(const vec3& a){ return (float)sqrt(a.x*a.x + a.y*a.y + a.z*a.z); }
  void print() { printf("(%f, %f, %f)", x,y,z); }
};


float noise3v(const vec3& p)
{
  const float mult = 30.0f;

  return noise3(mult*p.x,mult* p.y, mult*p.z);
}
template<class t> t max(const t& a, const t& b) 
{
  return a > b ? a: b;
}

template<class t> t min(const t& a, const t& b) 
{
  return a < b ? a: b;
}

inline float smoothstep(float a, float b,float t)
{
	if(t < a) return 0;
	if(t > b) return 1;
	float mu;
	t = (t-a)/(b-a);
	return (t) * (t) * (3 - 2*(t)) ;
}
inline float step(float a,float t)
{
	if(t <= a) return 0;
	return 1;
}

inline float pulse(float a,float b,float t)
{
	return step(a,t) - step(b,t);
}

inline float clamp(float a,float b,float t)
{
	return min(max(t,a),b);
}

inline float _frac(float a)
{
	return a - (int)a;//(int)(a-0.5);
}
inline float mod(float a,float t)
{
	return _frac(t/a);
}


struct sphere 
{
  vec3 pos; 
  float r ;
};


sphere sph[2];  

float isect (const vec3 p,const vec3 rd,vec3* normal)
{
  float t = 999.9f,tnow,b,disc;
  for (int i=0; i<2; i++) 
  { 
    tnow = 9999.9;           
    vec3 sd = sph[i].pos - p;    
    b = dot (rd,sd);
    disc = b*b + sph[i].r - dot ( sd,sd );
    if (disc>0.0) 
      tnow = b - sqrt(disc);
    if ((tnow>0.0001)&&(t>tnow)) 
    {
      t = tnow; 
      *normal = normalize(sph[i].pos - (p + t*rd));
    }
  }
  return t;
}
///distance from point to shpere
float d_shpere(const sphere& sp, vec3 pos)
{
  return distance(sp.pos - pos) - sp.r;
}

float distanceToBox(const vec3& p, float a, float b, float c )
{
  const float dx = max( fabsf( p.x )-a, 0.0f );
  const float dy = max( fabsf( p.y )-b, 0.0f );
  const float dz = max( fabsf( p.z )-c, 0.0f );

  return( sqrtf(dx*dx + dy*dy + dz*dz) );
   
}


float scene(const vec3& pp)
{
const int NN = 4;
 vec3 p  = pp;
//  printf("%f,",pp.x);
//  printf("%f,%f\n",pp.x, _frac(pp.x));
//
 const float s = 0.5f;
 const float h = 0.1f;
 float m = pp.z;// -  0.1f*(1.0f - step(h,mod(s,fabs(pp.x)))*step(h,mod(s, fabs(p.y))));

 m = min(m, distanceToBox(pp - vec3(0.0f,0.0f,0.52f), 0.5f, 0.5f, 0.5f));
 return m;

  
 for(int i = 0; i < NN; ++i)
 {
 	for(int j = 0; j < NN ; ++j)
	{
		m = min(m, min(distanceToBox(p, 0.5f, 0.5f, 0.5f),d_shpere(sph[0], p + vec3(float(i),float(j),0.0f))));
	}
 }
 return m;
 return  min(p.z,min(d_shpere(sph[0], p), d_shpere(sph[0], p + vec3(3.0f, 0.0f,0.0f))));
}

float ao(const vec3& p, const vec3& normal)
{

  const int N = 5;
  const float eps = 0.05f;
  int ocluded = 0;
  float term = 0.0f;
  for(int i = 0; i < N; ++i)
  {
    float delta = i*eps;
    vec3 pos = i*eps*normal;
    float d = scene(p + pos);
    //printf("distance: %f: %f\n", d, distance(pos));
    term += max((delta - d),0.0f)/float(1<<(i));
  }
  float a = 1.0 - term;
  return a*a*a;
}

float shadow(const vec3 p, const vec3 lpos)
{
	const int N = 4;
	float d = distance(lpos - p);
	vec3 dir = normalize(lpos - p);
	//d /= float(N);
	float s = 1.0f;
	for(int i = 0; i<N; ++i)
	{
		float delta = d/float(2<<(N - i -1 ));
		float dist = scene(p + delta * dir);
		//if(fabs(dist - delta) > 0.0005f)
		if(dist - delta < 0.0025f)
			s-= dist/(d*float(1 << ( i)));
	
	}
	//printf("%f // %f\n", s, d);
  return max(1.0f - s, 0.0f);
	//return float(s)/float(d);
}

vec3 raycast(const vec3 p, const vec3 dir, vec3* normal, int depth = 0)
{
  //float dist = d_shpere(sph[0], p);
  float dist = scene(p);
  if (fabs(dist) < 0.005f)
  {
    
    const float eps = 0.005f;
    *normal = normalize(vec3( 
        scene(p + vec3(eps, 0.0f, 0.0f)) - scene(p - vec3(eps, 0.0f, 0.0f)),
        scene(p + vec3(0.0f, eps, 0.0f)) - scene(p - vec3(0.0f, eps, 0.0f)),
        scene(p + vec3(0.0f, 0.0f, eps)) - scene(p - vec3(0.0f, 0.0f, eps))
        ));
     vec3 bump = normalize(vec3( 
        noise3v(p + vec3(eps, 0.0f, 0.0f)) - noise3v(p - vec3(eps, 0.0f, 0.0f)),
        noise3v(p + vec3(0.0f, eps, 0.0f)) - noise3v(p - vec3(0.0f, eps, 0.0f)),
        noise3v(p + vec3(0.0f, 0.0f, eps)) - noise3v(p - vec3(0.0f, 0.0f, eps))
        ));
     //*normal = normalize(*normal + 0.3f*bump);

    return p;
  }
  else 
  {
    if(dist > 200.0f || depth > 200)
    {
      return p;
    }
    return raycast(p + dist*dir, dir, normal, depth + 1);
  }

}

int render(vec3 campos, const char* fname)
{
  int w = 500;
  int h = 400;
 
  byte* pixels = new byte[w*h*3];
  vec3 lookat;
  vec3 d = lookat - campos;
  vec3 r = normalize(cross(d, vec3(0.0f, 0.0f, 1.0f)));
  vec3 up = normalize(cross(r, d));
  campos.print();
  d.print();
  r.print();
  up.print();

  sph[0].pos = vec3(0.0f,0.0f,0.3f);
  sph[0].r = 0.3f;
  
  sph[1].pos = vec3(0.0f,2.0f,0.0f);
  sph[1].r =  0.4f;

  float aspect = 4.0f/3.0f;
  float near = 0.010f;
  float fov = 45*3.14f/180.0f;
  float ww = 2.0f*near/tan(fov*0.5f);
  float hh = ww/aspect;

  for(int i = 0; i < w; ++i)
  {
    for(int j = 0; j < h; ++j)
    {
      float x = float(i-(w>>1))/float(w);
      float y = float(j-(h>>1))/float(h);

      vec3 dir = normalize(near*d + ww*x*r + hh*y*up);
      vec3 n;
      vec3 inter = raycast(campos, dir, &n);
      float f = distance(campos - inter);
      vec3 lpos  = vec3(-5.0f,5.0f,5.0f);
      if( f > 0.0f)
      {
        float a = ao(inter, n);
        float ff  = max(dot(normalize(lpos -inter ), n), 0.0f);
        float s = shadow(inter, lpos);
        f = (s*a*a*0.8 + 0.1*ff);
        //f = a*0.8f + 0.1f*ff;
        f = min(f,1.0f);
        pixels[(j*w + i)*3] = 255 *f;
        pixels[(j*w + i)*3 + 1] = 255*f;
        pixels[(j*w + i)*3 + 2] = 255*f;
      }
      else
      {
        pixels[(j*w + i)*3] = 0;
        pixels[(j*w + i)*3 + 1] = 0;
        pixels[(j*w + i)*3 + 2] = 0;
      }
#if 0 
      float t = isect(campos, dir, &n);
      if( t < 800.0f)
      {
        //n.print();
        float f = max(dot(normalize(vec3(-1.0f,-1.0f,-1.0f)), n), 0.0f);
        pixels[(j*w + i)*3] = 255 *f;
        pixels[(j*w + i)*3 + 1] = 255*f;
        pixels[(j*w + i)*3 + 2] = 255*f;
      }
      else
      {
        pixels[(j*w + i)*3] = 0;
        pixels[(j*w + i)*3 + 1] = 0;
        pixels[(j*w + i)*3 + 2] = 0;
      }
#endif

    }
  }
  save_tga(pixels, w, h,fname);
}
#define PI 3.141516f
int main()
{
  char fname[256];
  int i = 45; 
  //for(int i = 0; i<1; ++i)
  {
    //sprintf(fname,"m\\movie_%03d.tga", i);
    vec3 campos(-100.0f*cos(i*PI/180.0f), -100.0f*sin(i*PI/180.0f), 50.0f);
    render(campos,"shot2.tga");

  }
}

#if 0 
static const char vsh_2d[] = \
"uniform vec4 fpar[4];"
"void main(void)"
"{"
    "gl_Position=gl_Vertex;"
    "vec3 d=normalize(fpar[1].xyz-fpar[0].xyz);"
    "vec3 r=normalize(cross(d,vec3(0.0,1.0,0.0)));"
    "vec3 u=cross(r,d);"
    "vec3 e=vec3(gl_Vertex.x*1.333,gl_Vertex.y,.75);"   //    eye space ray
    "gl_TexCoord[0].xyz=mat3(r,u,d)*e;"                 //  world space ray
    "gl_TexCoord[1]=vec4(.5)+gl_Vertex*.5;"             // screen space coordinate
"}";

// camera position
    fparams[ 0] = 2.0f*sinf(1.0f*t+0.1f);
    fparams[ 1] = 0.0f;
    fparams[ 2] = 2.0f*cosf(1.0f*t);
    // camera target
    fparams[ 4] = 0.0f;
    fparams[ 5] = 0.0f;
    fparams[ 6] = 0.0f;
    // sphere
    fparams[ 8] = 0.0f;
    fparams[ 9] = 0.0f;
    fparams[10] = 0.0f;
    fparams[11] = 1.0f;

#endif 
