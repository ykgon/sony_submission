#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define M_PI 3.14159265358979323846         
int N;                                     


typedef struct{
    double  **r;                                //粒子位置
    double  *u;                                 //関数値
    double  *u_analytical;                      //解析解
    double  *error_laplace;                     //誤差の計算
    int     *particle_type;                     //境界粒子の分類

    int      n;                                 //一辺の粒子数                          
    double   dis;                               //初期粒子間距離        
    double   h;                                 //影響半径    
    double   L2error_laplace_boundary;          
    double   L2error_laplace_inner;                 
}CALC_PARTICLES;


typedef struct{
    double **mat;
    double *rhs;
} LIN_SYS;


void memory_allocation(
    CALC_PARTICLES      *cp,
    LIN_SYS             *ls,
    int dof                 )
{
    cp->r                   = (double**)calloc(dof, sizeof(double));
    for(int i=0; i<dof; i++){
        cp->r[i] = (double*)calloc(dof, sizeof(double));
    }
    
    cp->particle_type       = (int *)calloc(dof,sizeof(int));
    cp->u                   = (double*)calloc(dof, sizeof(double));
    cp->u_analytical        = (double*)calloc(dof, sizeof(double));
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
		int n   )
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


double weight_func_x(
            double q,
            double h,
            double x_unit   )
{
    double k = 1.0 / 3.0;
    double l = 2.0 / 3.0;
    double a = -76545.0 / (478.0*M_PI*pow(h,3.0));

    if(q<k){
        return(a*(pow((1-q),4.0) - 6.0*pow((l-q),4.0) + 15*pow((k-q),4.0)) * x_unit);
    }
    else if(q<l){
        return(a*(pow((1-q),4.0) - 6.0*pow((l-q),4.0)) * x_unit);
    }
    else if(q<1){
        return(a*pow((1-q),4.0) * x_unit);
    }
    else{
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

    if(q<k){
        return(a*(pow((1-q),4.0) - 6.0*pow((l-q),4.0) + 15*pow((k-q),4.0)) * y_unit);
    }
    else if(q<l){
        return(a*(pow((1-q),4.0) - 6.0*pow((l-q),4.0)) * y_unit);
    }
    else if(q<1){
        return(a*pow((1-q),4.0) * y_unit);
    }
    else{
        return(0);
    } 
}


double calc_const(
    double norm,
    double h,
    double dis      )
{
    double nyu = 0.0001 * pow(h, 2.0);
    double a = 2.0 * dis * dis;
    double b =  1.0 / (pow(norm, 2.0) + nyu);
    return(a * b);
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


void set_matrix(
    CALC_PARTICLES  *cp,
    LIN_SYS         *ls,
    int dof             )
{
    double w_x, w_y, xij, yij, q, norm, x_unit, y_unit;
    
    for(int i=0; i<dof; i++){                                                              
        for(int j=0; j<dof; j++){
            xij = cp->r[i][0] - cp->r[j][0];                                           
            yij = cp->r[i][1] - cp->r[j][1];                                            
            norm = sqrt(((xij*xij) + (yij*yij)));                                       
            if(0<norm && norm<cp->h){
                x_unit = xij/norm;
                y_unit = yij/norm;
                q = norm/cp->h;
                w_x = weight_func_x(q, cp->h, x_unit);                                 
                w_y = weight_func_y(q, cp->h, y_unit);                                
                ls->mat[i][j] = calc_const(norm, cp->h, cp->dis) * calc_product(w_x, w_y, xij, yij);                                           
                ls->mat[i][i] -= calc_const(norm, cp->h, cp->dis) * calc_product(w_x, w_y, xij, yij);                                               
            }    
        }
        ls->rhs[i] = 0.0;                                       
    }
}


void calc_u_analytical(
    CALC_PARTICLES *cp,
    int             dof )
{
    for(int i=0; i<dof; i++){
        double a = 1 / sinh(M_PI);
        cp->u_analytical[i] = a * sin(cp->r[i][0]) * sinh(cp->r[i][1]);
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
            if(cp->particle_type[k] == 1){ 
                //右辺ベクトルからマトリクスの境界粒子の列を引く
                for(int l=0; l<dof; l++){
                    if(cp->n-1 == j){
                        ls->rhs[l] -= ls->mat[l][k] * sin(cp->r[k][0]);
                    }
                    else{
                        ls->rhs[l] -= ls->mat[l][k] * 0.0;
                    }
                    if(k==l){
                        ls->mat[k][l] = 1.0;
                    }
                    else{
                        ls->mat[k][l] = 0.0;
                        ls->mat[l][k] = 0.0;
                    }
                }
                if(cp->n-1 == j){
                    ls->rhs[k] = sin(cp->r[k][0]);
                }
                else{
                    ls->rhs[k] = 0.0;
                }
            }
        }
    }
}



/*ガウスの消去法*/
void solve_matrix_GE(
    CALC_PARTICLES  *cp, 
    LIN_SYS         *ls,
    int dof              )
{
    int k, m;
    double p, escape;

    /*前進消去*/
    for(int i=0; i<dof-1; i++){
        /*絶対値が最大の行を選択*/
        m = i;
        for(int j=i+1; j<dof; j++){
            if(fabs(ls->mat[j][i]) > fabs(ls->mat[m][i])){
                m = j;
            }
        }

        /*i行目とm行目の入れ替え*/
        for(int j=0; j<dof; j++){
            escape = ls->mat[i][j];
            ls->mat[i][j] = ls->mat[m][j];
            ls->mat[m][j] = escape;
        }
        escape = ls->rhs[i];
        ls->rhs[i] = ls->rhs[m];
        ls->rhs[m] = escape;

        for(int j=i+1; j<dof; j++){
            p = ls->mat[j][i] / ls->mat[i][i];
            for(k=i; k<dof; k++){
                ls->mat[j][k] -= p * ls->mat[i][k]; 
            }
            ls->rhs[j] -= p * ls->rhs[i];
        }
    }

    /*後退代入*/
    for(int i=dof-1; i>=0; i--){
        cp->u[i] = ls->rhs[i] / ls->mat[i][i];
        for(int j=i+1; j<dof; j++){
            cp->u[i] -= (ls->mat[i][j] / ls->mat[i][i]) * cp->u[j];
        }
    }
}



void calc_L2error(
        CALC_PARTICLES *cp,
        int dof             )
{
    double sum1_boundary=0.0, sum2_boundary=0.0, sum1_inner=0.0, sum2_inner=0.0;

    for(int i=0; i<dof; i++){
        if((cp->r[i][0]<=cp->h) || (cp->r[i][1]<=cp->h) || (cp->r[i][0]>=1.0-cp->h) || (cp->r[i][1]>=1.0-cp->h)){
             sum1_boundary += pow((cp->u[i] - cp->u_analytical[i]), 2.0);
             sum2_boundary += pow(cp->u_analytical[i], 2.0);
        }
        else{
            sum1_inner += pow((cp->u[i] - cp->u_analytical[i]), 2.0);
            sum2_inner += pow(cp->u_analytical[i], 2.0);
        }
    }
    cp->L2error_laplace_boundary = sqrt(sum1_boundary)/sqrt(sum2_boundary);
    cp->L2error_laplace_inner = sqrt(sum1_inner)/sqrt(sum2_inner);
    printf("L2 error of u_boundary = %.15lf, L2 error of u_inner = %.15lf\n",cp->L2error_laplace_boundary, cp->L2error_laplace_inner);
}


void write_vtu(
    CALC_PARTICLES *cp,
    int dof             )
{
    FILE *file;

    file = fopen("laplace_solver.vtu","w");

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
    fprintf(file,"\t\t\t</DataArray>\n");
    fprintf(file,"\t\t\t<DataArray NumberOfComponents=\"1\" type=\"Float32\" Name=\"Analytical_Solution\" format=\"ascii\">\n");
    for(int i=0; i<dof; i++){
        fprintf(file,"\t\t\t\t%lf\n",cp->u_analytical[i]);
    }
    fprintf(file,"\t\t\t</DataArray>\n");
    fprintf(file,"\t\t\t<DataArray NumberOfComponents=\"1\" type=\"Float32\" Name=\"Error\" format=\"ascii\">\n");
    for(int i=0; i<dof; i++){
        fprintf(file,"\t\t\t\t%lf\n",fabs(cp->u[i] - cp->u_analytical[i]));
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
    free(cp->u_analytical);
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
    
    //定数を構造体に挿入
    cp.n = N;
    cp.dis = M_PI / (cp.n-1);          
    cp.h = 3.2 * cp.dis;                
    printf("dis = %f\n", cp.dis);
    printf("h = %f\n", cp.h);

    memory_allocation(&cp, &ls, N*N);
    
    set_particles(&cp);
    calc_u_analytical(&cp, N*N);
    
    set_matrix(&cp, &ls, N*N);
    set_bc(&cp, &ls, N*N);
    solve_matrix_GE(&cp, &ls, N*N);
    calc_L2error(&cp, N*N);
    
    write_vtu(&cp, N*N);

    memory_free(&cp, &ls, N*N);
    
    return 0;

}
