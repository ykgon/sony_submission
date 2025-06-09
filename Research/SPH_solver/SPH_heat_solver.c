#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define M_PI 3.14159265358979323846         
int N;                                      
const double DT = 0.1; 
const int N_STEP = 100; 
const double ALPHA = 0.0112;             


typedef struct{
    double  **r;                                
    double  *u;                                 

    double  *error_laplace;                     
    int     *particle_type;                     
    double  *u_initial;                         

    int      n;                                                      
    double   dis;                                    
    double   h;                                 
    double   L2error_laplace_boundary;         
    double   L2error_laplace_inner;                   
}CALC_PARTICLES;


typedef struct{
    double **mat;
    double *rhs;
} LIN_SYS;


void memory_allocation(
    CALC_PARTICLES  *cp,
    LIN_SYS         *ls,
    int dof              )
{
    cp->r                   = (double**)calloc(dof, sizeof(double));
    for(int i=0; i<dof; i++){
        cp->r[i] = (double*)calloc(dof, sizeof(double));
    }
    
    cp->particle_type       = (int *)calloc(dof,sizeof(int));
    cp->u                   = (double*)calloc(dof, sizeof(double));
    cp->u_initial           = (double*)calloc(dof, sizeof(double));
    cp->error_laplace       = (double*)calloc(dof, sizeof(double));

    ls->mat                 = (double**)calloc(dof, sizeof(double));
    for(int i=0; i<dof; i++){
        ls->mat[i] = (double*)calloc(dof, sizeof(double));
    }

    ls->rhs                 = (double*)calloc(dof, sizeof(double));
}


int set_index(
		int i,
		int j,
		int n)
{
	return ( n*j + i );
}


void set_particles(
    CALC_PARTICLES       *cp)
{
    int k = 0;
    for(int j=0; j<cp->n; j++){
        for(int i=0; i<cp->n; i++){
            int k = set_index(i, j, cp->n);
            cp->r[k][0] = cp->dis * (double)(i);                      
            cp->r[k][1] = cp->dis * (double)(j);
            if(i==0 || cp->n-1==i || j==0 || cp->n-1==j){
                cp->particle_type[k] = 1;
            } else{
                cp->particle_type[k] = 0;
            }
        }
    }
}


double GetRandom(double min, double max)
{
    srand(time(NULL));
    return min + (double)(rand() * (max - min + 1.0) / (1.0 + RAND_MAX));
}


void set_u_initial(
    CALC_PARTICLES *cp,
    int dof             )
{
    double xij, yij, norm;
    double d = GetRandom(3.0, 5.0);        //熱源の半径
    printf("d = %f\n", d);
    int l;                         
    double t = GetRandom(0.0, 100.0);   
    printf("t = %f\n", t);
    double selected_i = GetRandom(d, N-d);
    double selected_j = GetRandom(d, N-d);
    l = set_index(selected_i, selected_j, cp->n);
    printf("l = %d\n", l);
    for(int i=0; i<N; i++){                                                              
        for(int j=0; j<N; j++){
            int k = set_index(i, j, cp->n);
            /*ピポッド粒子を決定*/
            xij = cp->r[k][0] - cp->r[l][0];                                          
            yij = cp->r[k][1] - cp->r[l][1];                                            
            norm = sqrt(((xij*xij) + (yij*yij))); 
            //printf("norm = %f\n", norm);                                       
            if(0.0<=norm && norm<=(d*cp->dis)){
                cp->u[k] = t;           
                cp->u[l] = t;
            }
        }
    }
}


double weight_func_x(
            double q,
            double h,   
            double x_unit   )
{
    double k = 1.0 / 3.0;
    double l = 2.0 / 3.0;
    double a = -76545.0 / (478.0*M_PI*pow(h,3.0));

    if(0<=q<k){
        return(a*(pow((1-q),4.0) - 6.0*pow((l-q),4.0) + 15*pow((k-q),4.0)) * x_unit);
    }
    if(k<=q<l){
        return(a*(pow((1-q),4.0) - 6.0*pow((l-q),4.0)) * x_unit);
    }
    if(l<=q<1){
        return(a*pow((1-q),4.0) * x_unit);
    }
    if(1<=q){
        return(0);
    } 
}


double weight_func_y(
            double q,
            double h,   
            double y_unit   )
{
    double k = 1.0 / 3.0;
    double l = 2.0 / 3.0;
    double a = -76545.0 / (478.0*M_PI*pow(h,3.0));

    if(0<=q<k){
        return(a*(pow((1-q),4.0) - 6.0*pow((l-q),4.0) + 15*pow((k-q),4.0)) * y_unit);
    }
    if(k<=q<l){
        return(a*(pow((1-q),4.0) - 6.0*pow((l-q),4.0)) * y_unit);
    }
    if(l<=q<1){
        return(a*pow((1-q),4.0) * y_unit);
    }
    if(1<=q){
        return(0);
    } 
}


double calc_const(
    double norm,
    double dis  )
{
    double a = dis * dis;
    double b =  1.0 / (pow(norm, 2.0) );
    return(a * b * DT);
}


double calc_product(
    double w_x,
    double w_y,
    double x,
    double y        )
{
    double x_product, y_product;
    x_product = -x * w_x;                                               
    y_product = -y * w_y;                                                
    return(x_product + y_product);
}


void set_mat(
    CALC_PARTICLES  *cp,
    LIN_SYS         *ls,
    int             dof,
    double          alpha   )
{
    double w_x, w_y, xij, yij, q, norm, x_unit, y_unit;
     for(int i=0; i<dof; i++){                                                              
        for(int j=0; j<dof; j++){
            ls->mat[i][j] = 0.0;
        }
     }
    for(int i=0; i<dof; i++){                                                              
        for(int j=0; j<dof; j++){
            xij = cp->r[i][0] - cp->r[j][0];                                           
            yij = cp->r[i][1] - cp->r[j][1];                                            
            norm = sqrt(((xij*xij) + (yij*yij)));                                      
            if(0.0<=norm && norm<=cp->h){
                x_unit = xij/norm;
                y_unit = yij/norm;
                q = norm/cp->h;
                w_x = weight_func_x(q, cp->h, x_unit);                                 
                w_y = weight_func_y(q, cp->h, y_unit);                                  
                if(j!=i){
                    ls->mat[i][j] = -(2.0*alpha) * calc_const(norm, cp->dis) * calc_product(w_x, w_y, xij, yij);     
                    ls->mat[i][i] += calc_const(norm, cp->dis) * calc_product(w_x, w_y, xij, yij);                                      
                }
            }    
        }
        ls->mat[i][i] *= (2.0*alpha);
        ls->mat[i][i] += 1.0;   
        if(cp->particle_type[i] == 0){
            ls->rhs[i] = cp->u[i];
        }                                       
    }  
}


void set_bc(
    CALC_PARTICLES  *cp,
    LIN_SYS         *ls,
    int             dof )
{
    for(int j=0; j<cp->n; j++) {
		for(int i=0; i<cp->n; i++) {
			int k = set_index(i, j, cp->n);
            if( i == 0 || i == cp->n-1 
					|| j == 0 || j == cp->n-1 ) {

				for(int l=0; l<dof; l++) {
					ls->mat[k][l] = 0.0;
					ls->mat[l][k] = 0.0;
				}

				ls->mat[k][k] = 1.0;
				ls->rhs[k]    = 0.0;
			}
        }
    }
}


static const int MAX_ITERATIONS = 10000;
static const double EPSILON = 0.0000001;
void solve_mat_GS(
		double* sol,
		double** mat,
		double*  rhs, 
		int length)
{
	for(int l=0; l<MAX_ITERATIONS; l++) {

		for(int i=0; i<length; i++) {
			sol[i] = rhs[i];
			for(int j=0; j<length; j++) {
				if(i != j) {
					sol[i] -= mat[i][j]*sol[j];
				}
			}
			sol[i] /= mat[i][i];
		}

		double residual = 0.0;
		for(int i=0; i<length; i++) {
			double r_i = -rhs[i];
			for(int j=0; j<length; j++) {
				r_i += mat[i][j]*sol[j];
			}
			residual += r_i*r_i;
		}
		residual = sqrt(residual);
		printf("GS_loop %d: %e\n", l, residual);
		if(residual < EPSILON) return;
	}
}


void write_vtu(
    CALC_PARTICLES *cp,
    int file_num,
    int dof             )
{
    FILE *file;
	char filename[10000];
	snprintf(filename, 10000, "./data_5/out_%06d.vtu", file_num);
    file = fopen(filename,"w");
    if (file == NULL) {
    perror("Error opening file");
    return;
    }

    fprintf(file,"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(file,"<VTKFile xmlns=\"VTK\" byte_order=\"LittleEndian\" version=\"0.1\" type=\"UnstructuredGrid\">\n");
    fprintf(file,"<UnstructuredGrid>\n");
    fprintf(file,"\t<Piece NumberOfCells=\"%d\" NumberOfPoints=\"%d\">\n",dof,dof);
    fprintf(file,"\n\t\t<Points>\n");
    fprintf(file,"\t\t\t<DataArray NumberOfComponents=\"3\" type=\"Float32\" Name=\"Position\" format=\"ascii\">\n");
    for(int j=0; j<cp->n; j++){
        for(int i=0; i<cp->n; i++){
            int k = set_index(i, j, cp->n);
            fprintf(file,"\t\t\t\t%lf %lf %lf\n",cp->r[k][0], cp->r[k][1], 0.0);
        }
    }
    fprintf(file,"\t\t\t</DataArray>\n\t\t</Points>\n");
    fprintf(file,"\n\t\t<PointData>\n");
    fprintf(file,"\t\t\t<DataArray NumberOfComponents=\"1\" type=\"Float32\" Name=\"Numerical_Solution\" format=\"ascii\">\n");
    for(int i=0;i<dof;i++){
        fprintf(file,"\t\t\t\t%lf\n",cp->u[i]);
    }
    fprintf(file,"\t\t\t</DataArray>\n\t\t</PointData>\n");
    fprintf(file,"\n\t\t<Cells>\n");
    fprintf(file,"\t\t\t<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
    for(int i=0;i<dof;i++){
        fprintf(file,"\t\t\t\t%d\n",i);
    }
    fprintf(file,"\t\t\t</DataArray>\n");
    fprintf(file,"\t\t\t<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
    for(int i=0;i<dof;i++){
        fprintf(file,"\t\t\t\t%d\n",i+1);
    }
    fprintf(file,"\t\t\t</DataArray>\n");
    fprintf(file,"\t\t\t<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
    for(int i=0;i<dof;i++){
        fprintf(file,"\t\t\t\t%d\n",1);
    }
    fprintf(file,"\t\t\t</DataArray>\n");
    fprintf(file,"\t\t</Cells>\n\n\t</Piece>\n</UnstructuredGrid>\n</VTKFile>");
    fclose(file);
}


void memory_free(
    CALC_PARTICLES  *cp,
    LIN_SYS         *ls,
    int dof             )
{
    for(int i=0; i<dof; i++){
        free(cp->r[i]);
    }
    free(cp->r);

    free(cp->particle_type);
    free(cp->u);
    free(cp->u_initial);
    free(cp->error_laplace);

    for(int i=0; i<dof; i++){
        free(ls->mat[i]);
    }
    free(ls->mat);
    
    free(ls->rhs);
}


int main(void)
{
    CALC_PARTICLES cp;
    LIN_SYS ls;  

    printf("n = ");
    scanf("%d", &N);
    
    cp.n = N;
    cp.dis = 1.0 / (cp.n-1);       
    cp.h = 3.2 * cp.dis;                
    printf("dis = %f\n", cp.dis);
    printf("h = %f\n", cp.h);

    memory_allocation(&cp, &ls, N*N);

    set_particles(&cp);
    set_u_initial(&cp, N*N);

    for(int n=0; n<N_STEP; n++) {
		printf("------ %d STEP -----\n", n);
        set_mat(&cp, &ls, N*N, ALPHA);
        set_bc(&cp, &ls, N*N);
       	solve_mat_GS(cp.u, ls.mat, ls.rhs, N*N);
        write_vtu(&cp, n, N*N);
    }
      
    memory_free(&cp, &ls, N*N);
    
    return 0;
}