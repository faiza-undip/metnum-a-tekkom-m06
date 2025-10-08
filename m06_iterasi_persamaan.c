// spnl.c — Sistem Persamaan Nonlinear (IT: Jacobi/Seidel, Newton-Raphson, Broyden-Secant)
// Persoalan: f1(x,y)=x^2+xy-10=0, f2(x,y)=y+3xy^2-57=0
// NIM: 21120123140056  -> NIMx = 56 % 4 = 0  -> IT default: g1A & g2A
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

typedef struct
{
    double x, y, dx, dy;
    int iter;
    int ok;
} Result;

static inline double f1(double x, double y) { return x * x + x * y - 10.0; }
static inline double f2(double x, double y) { return y + 3.0 * x * y * y - 57.0; }

// ---------- Fixed-Point Iteration (g-sets) ----------
// Set A (dari contoh slide yang divergen):
//   g1A = (10 - x^2)/y
//   g2A = 57 - 3*x*y^2
static inline double g1A(double x, double y) { return (10.0 - x * x) / y; }
// Jacobi pakai x_old; Seidel pakai x_new — kita handle di pemanggil
static inline double g2A(double x_used, double y) { return 57.0 - 3.0 * x_used * y * y; }

// Set B (yang konvergen di slide):
//   g1B = sqrt(10 - x*y)
//   g2B = sqrt( (57 - y) / (3*x_used) )
static inline double g1B(double x, double y)
{
    double t = 10.0 - x * y;
    return (t <= 0) ? NAN : sqrt(t);
}
static inline double g2B(double x_used, double y)
{
    double den = 3.0 * x_used;
    double num = 57.0 - y;
    if (den <= 0)
        return NAN;
    double t = num / den;
    return (t < 0) ? NAN : sqrt(t);
}

// IT – Jacobi
Result it_jacobi(double x0, double y0, double eps, int maxit,
                 double (*g1)(double, double),
                 double (*g2)(double, double))
{
    Result r = {x0, y0, 0, 0, 0, 0};
    for (int k = 1; k <= maxit; k++)
    {
        double xn = g1(r.x, r.y);
        double yn = g2(r.x, r.y); // Jacobi: pakai x_old
        if (!isfinite(xn) || !isfinite(yn))
        {
            r.ok = 0;
            r.iter = k;
            return r;
        }
        r.dx = fabs(xn - r.x);
        r.dy = fabs(yn - r.y);
        r.x = xn;
        r.y = yn;
        r.iter = k;
        if (r.dx < eps && r.dy < eps)
        {
            r.ok = 1;
            return r;
        }
    }
    r.ok = 0;
    return r;
}

// IT – Seidel
Result it_seidel(double x0, double y0, double eps, int maxit,
                 double (*g1)(double, double),
                 double (*g2)(double, double))
{
    Result r = {x0, y0, 0, 0, 0, 0};
    for (int k = 1; k <= maxit; k++)
    {
        double xn = g1(r.x, r.y);
        double yn = g2(xn, r.y); // Seidel: x baru langsung dipakai
        if (!isfinite(xn) || !isfinite(yn))
        {
            r.ok = 0;
            r.iter = k;
            return r;
        }
        r.dx = fabs(xn - r.x);
        r.dy = fabs(yn - r.y);
        r.x = xn;
        r.y = yn;
        r.iter = k;
        if (r.dx < eps && r.dy < eps)
        {
            r.ok = 1;
            return r;
        }
    }
    r.ok = 0;
    return r;
}

// ---------- Newton–Raphson untuk sistem 2 variabel ----------
Result newton(double x0, double y0, double eps, int maxit)
{
    Result r = {x0, y0, 0, 0, 0, 0};
    for (int k = 1; k <= maxit; k++)
    {
        double u = f1(r.x, r.y), v = f2(r.x, r.y);
        double ux = 2.0 * r.x + r.y;
        double uy = r.x;
        double vx = 3.0 * r.y * r.y;
        double vy = 1.0 + 6.0 * r.x * r.y;
        double det = ux * vy - uy * vx;
        if (fabs(det) < 1e-15)
        {
            r.ok = 0;
            r.iter = k;
            return r;
        }
        double dx = (u * vy - v * uy) / det; // ini sebenarnya +, tapi kita update x - dx
        double dy = (v * ux - u * vx) / det;
        double xn = r.x - dx;
        double yn = r.y - dy;
        r.dx = fabs(xn - r.x);
        r.dy = fabs(yn - r.y);
        r.x = xn;
        r.y = yn;
        r.iter = k;
        if (r.dx < eps && r.dy < eps)
        {
            r.ok = 1;
            return r;
        }
    }
    r.ok = 0;
    return r;
}

// ---------- "Secant" multivariabel: Broyden (good) ----------
Result broyden(double x0, double y0, double eps, int maxit)
{
    // Mulai dengan Jacobian identitas untuk aproksimasi awal inverse
    double Binv[2][2] = {{1, 0}, {0, 1}}; // aproksimasi inverse Jacobian
    Result r = {x0, y0, 0, 0, 0, 0};
    double fx = f1(r.x, r.y), fy = f2(r.x, r.y);
    for (int k = 1; k <= maxit; k++)
    {
        // s = -B^{-1} F
        double sx = -(Binv[0][0] * fx + Binv[0][1] * fy);
        double sy = -(Binv[1][0] * fx + Binv[1][1] * fy);
        double xn = r.x + sx, yn = r.y + sy;
        double fxn = f1(xn, yn), fyn = f2(xn, yn);
        // cek konvergensi
        r.dx = fabs(xn - r.x);
        r.dy = fabs(yn - r.y);
        r.x = xn;
        r.y = yn;
        r.iter = k;
        if (r.dx < eps && r.dy < eps && fabs(fxn) < 1e-8 && fabs(fyn) < 1e-8)
        {
            r.ok = 1;
            return r;
        }
        // y = F_new - F_old
        double yv[2] = {fxn - fx, fyn - fy};
        // u = B^{-1} y
        double ux = Binv[0][0] * yv[0] + Binv[0][1] * yv[1];
        double uy = Binv[1][0] * yv[0] + Binv[1][1] * yv[1];
        // denom = s^T u
        double denom = sx * ux + sy * uy;
        if (fabs(denom) < 1e-15)
        {
            r.ok = 0;
            return r;
        }
        // Rank-1 update (good Broyden) untuk B^{-1}
        double coef = 1.0 / denom;
        Binv[0][0] += (sx - ux) * sx * coef;
        Binv[0][1] += (sx - ux) * sy * coef;
        Binv[1][0] += (sy - uy) * sx * coef;
        Binv[1][1] += (sy - uy) * sy * coef;
        fx = fxn;
        fy = fyn;
        if (!isfinite(Binv[0][0]) || !isfinite(Binv[1][1]))
        {
            r.ok = 0;
            return r;
        }
    }
    r.ok = 0;
    return r;
}

static void print_header(const char *title)
{
    printf("\n== %s ==\n", title);
    printf(" iter           x               y           |dx|           |dy|\n");
    printf("-----------------------------------------------------------------\n");
}

static void demo_iterative_table(double x0, double y0, double eps, int maxit,
                                 int seidel, int setAB)
{
    // pilih g
    double (*g1)(double, double) = setAB == 0 ? g1A : g1B;
    double (*g2)(double, double) = setAB == 0 ? g2A : g2B;

    const char *method = seidel ? "IT-Seidel" : "IT-Jacobi";
    const char *gset = setAB == 0 ? "g1A & g2A" : "g1B & g2B";
    char title[128];
    snprintf(title, sizeof(title), "%s (%s)", method, gset);
    print_header(title);

    double x = x0, y = y0;
    printf("%5d %14.6f %14.6f %12.6f %12.6f\n", 0, x, y, 0.0, 0.0);
    for (int k = 1; k <= maxit; k++)
    {
        double xn = g1(x, y);
        double yn = seidel ? g2(xn, y) : g2(x, y);
        double dx = fabs(xn - x), dy = fabs(yn - y);
        printf("%5d %14.6f %14.6f %12.6f %12.6f\n", k, xn, yn, dx, dy);
        x = xn;
        y = yn;
        if (!isfinite(x) || !isfinite(y))
        {
            printf("-> Divergen/NaN pada iterasi %d\n", k);
            return;
        }
        if (dx < eps && dy < eps)
        {
            printf("-> Konvergen dalam %d iterasi\n", k);
            return;
        }
    }
    printf("-> Belum konvergen (maks iter tercapai)\n");
}

int main(int argc, char **argv)
{
    double x0 = 1.5, y0 = 3.5, eps = 1e-6;
    int maxit = 200;
    if (argc >= 3)
    {
        x0 = atof(argv[1]);
        y0 = atof(argv[2]);
    }
    if (argc >= 4)
    {
        eps = atof(argv[3]);
    }
    if (argc >= 5)
    {
        maxit = atoi(argv[4]);
    }

    // NIMx untuk info saja
    int nim_last2 = 56; // dari NIM 21120123140056
    int NIMx = nim_last2 % 4;
    printf("NIM: 21120123140056  -> NIMx = %d\n", NIMx);
    printf("Start (x0,y0) = (%.6f, %.6f), eps = %.8f, maxIter = %d\n", x0, y0, eps, maxit);

    // 1) IT Jacobi & Seidel — default sesuai NIMx:
    // NIMx=0 -> gunakan g1A & g2A (kombinasi lain tetap disediakan di bawah)
    demo_iterative_table(x0, y0, eps, maxit, 0, 0); // Jacobi, Set A
    demo_iterative_table(x0, y0, eps, maxit, 1, 0); // Seidel, Set A

    // BONUS: Jalankan juga set yang konvergen (B) biar kelihatan bedanya
    demo_iterative_table(x0, y0, eps, maxit, 0, 1); // Jacobi, Set B
    demo_iterative_table(x0, y0, eps, maxit, 1, 1); // Seidel, Set B

    // 2) Newton–Raphson
    print_header("Newton-Raphson");
    Result rn = newton(x0, y0, eps, maxit);
    printf("%5d %14.6f %14.6f %12.6g %12.6g\n", rn.iter, rn.x, rn.y, rn.dx, rn.dy);
    printf("-> %s dalam %d iterasi; f1=%.3e, f2=%.3e\n",
           rn.ok ? "Konvergen" : "Gagal konvergen", rn.iter, f1(rn.x, rn.y), f2(rn.x, rn.y));

    // 3) Secant (Broyden)
    print_header("Secant (Broyden)");
    Result rb = broyden(x0, y0, eps, maxit);
    printf("%5d %14.6f %14.6f %12.6g %12.6g\n", rb.iter, rb.x, rb.y, rb.dx, rb.dy);
    printf("-> %s dalam %d iterasi; f1=%.3e, f2=%.3e\n",
           rb.ok ? "Konvergen" : "Gagal konvergen", rb.iter, f1(rb.x, rb.y), f2(rb.x, rb.y));

    return 0;
}
