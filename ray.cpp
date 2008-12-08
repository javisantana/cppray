#include <stdio.h>
#include <math.h>

float IntNoise(register int x)
{
    x = (x<<13)^x;
    return (((x * (x * x * 15731 + 789221) + 1376312589) & 0x7fffffff) / 2147483648.0);
}

float InterPol(float a, float b, float x)
{     //a altura 1a
    return a+(b-a)*x*x*(3-2*x);                //b altura 2a
}        

float _PerlinNoise(float x,float y,int width,int octaves,int seed)
{
           
           double a,b,valor = 0,freq,tam_pas,cachox,cachoy;
           int casilla,num_pasos,pasox,pasoy;
		   int s;
           int amplitud=256;                        //La amplitud es 128,64,32,16... para cada pasada
           int periodo=256;                         //El periodo es similar a la amplitud
           for (s=0;s<octaves;s++)
	   {
                amplitud>>=1;                      //Manera rápida de dividir entre 2
                periodo>>=1;
                freq=1/(float)periodo;             //Optimizacion para dividir 1 vez y multiplicar luego
                num_pasos=(int)(width*freq);         //Para el const que vimos en IntNoise
                pasox=(int)(x*freq);                 //Indices del vértice superior izquerda del cuadrado
                pasoy=(int)(y*freq);                 //en el que nos encontramos
                cachox=x*freq-pasox;               //frac_x y frac_y en el ejemplo
                cachoy=y*freq-pasoy;
                casilla=pasox+pasoy*num_pasos;     // índice final del IntNoise
                a=InterPol(IntNoise(casilla+seed),IntNoise(casilla+1+seed),cachox);
                b=InterPol(IntNoise(casilla+num_pasos+seed),IntNoise(casilla+1+num_pasos+seed),cachox);
                valor+=InterPol(a,b,cachoy)*amplitud;   //superposicion del valor final con anteriores
            }
          return valor;                           //seed es un numero que permite generar imagenes distintas
 }     

float noise2d(float x, float y)
{
	return _PerlinNoise(x,y, 256, 5, 10);
}
typedef unsigned char byte;
void save_tga(byte* pixels, int w, int h)
{
  FILE* shot;
  if((shot=fopen("shot.tga", "wb"))==0) return ;
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

template<class t> t max(const t& a, const t& b) 
{
  return a > b ? a: b;
}

template<class t> t min(const t& a, const t& b) 
{
  return a < b ? a: b;
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

float scene(const vec3& p)
{
const int NN = 4;
 float m = p.z;
 for(int i = 0; i < NN; ++i)
 {
 	for(int j = 0; j < NN ; ++j)
	{
		m = min(m, d_shpere(sph[0], p + vec3(float(i),float(j),0.0f)));
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
  return 1.0 - term;
}

float shadow(const vec3 p, const vec3 lpos)
{
	const int N = 5;
	float d = distance(lpos - p);
	vec3 dir = normalize(lpos - p);
	//d /= float(N);
	float s = 0.0f;
	for(int i = 0; i<N; ++i)
	{
		float delta = d/float(2<<(N - i -1 ));
		float dist = scene(p + delta * dir);
		if(fabs(dist - delta) > 0.01f)
			s+= d/float(1 << ( i));
	
	}
	//printf("%f // %f\n", s, d);
	return float(s)/float(d);
}

vec3 raycast(const vec3 p, const vec3 dir, vec3* normal, int depth = 0)
{
  //float dist = d_shpere(sph[0], p);
  const float eps = 0.01f;
  float dist = scene(p);
  if (fabs(dist) < 0.01f)
  {
    
    *normal = normalize(vec3( 
        scene(p + vec3(eps, 0.0f, 0.0f)) - scene(p - vec3(eps, 0.0f, 0.0f)),
        scene(p + vec3(0.0f, eps, 0.0f)) - scene(p - vec3(0.0f, eps, 0.0f)),
        scene(p + vec3(0.0f, 0.0f, eps)) - scene(p - vec3(0.0f, 0.0f, eps))
        ));

    return p;
  }
  else 
  {
    if(dist > 200.0f)
    {
      return p;
    }
    return raycast(p + dist*dir, dir, normal, depth + 1);
  }

}

int main()
{
  int w = 500;
  int h = 400;
  byte* pixels = new byte[w*h*3];
  vec3 campos(-100.0f, -100.0f, 50.0f);
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
    	float a = 1.0f;//ao(inter, n);
        float ff  = max(dot(normalize(lpos -inter ), n), 0.0f);
	float s = shadow(inter, lpos);
	f = s; //s*a*a*a *(0.2*ff);
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
  save_tga(pixels, w, h);
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
