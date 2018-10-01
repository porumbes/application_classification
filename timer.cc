#include <sys/time.h>
#include <time.h>

double my_timer() {
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  return ((double)tv.tv_sec + (double)0.000001 * (double)tv.tv_usec);
}
