#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  const char *name;
  size_t record_bytes;
  int source_text_exposed;
  int source_kv_exposed;
} profile_t;

static const profile_t PROFILES[] = {
    {"packet_2b_payload_5b_record", 5, 0, 0},
    {"query_aware_text_14b", 14, 1, 0},
    {"full_hidden_log_370b", 370, 1, 0},
    {"qjl_1bit_kv_floor_21504b", 21504, 0, 1},
    {"kivi_2bit_kv_floor_43008b", 43008, 0, 1},
};

static const size_t BATCHES[] = {1, 4, 16, 64, 256};

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int cmp_double(const void *lhs, const void *rhs) {
  double a = *(const double *)lhs;
  double b = *(const double *)rhs;
  return (a > b) - (a < b);
}

static double percentile(double *values, size_t count, double q) {
  double sorted[32];
  if (count > 32) {
    count = 32;
  }
  for (size_t i = 0; i < count; ++i) {
    sorted[i] = values[i];
  }
  qsort(sorted, count, sizeof(double), cmp_double);
  size_t index = (size_t)((count - 1) * q + 0.5);
  if (index >= count) {
    index = count - 1;
  }
  return sorted[index];
}

static size_t iterations_for(size_t total_bytes, uint64_t target_bytes, size_t min_iterations) {
  if (total_bytes == 0) {
    return min_iterations;
  }
  size_t iterations = (size_t)(target_bytes / total_bytes);
  if (iterations < min_iterations) {
    iterations = min_iterations;
  }
  return iterations;
}

static void fill_source(uint8_t *src, size_t total_bytes, size_t record_bytes) {
  for (size_t i = 0; i < total_bytes; ++i) {
    src[i] = (uint8_t)((i * 131u + record_bytes * 17u) & 0xffu);
  }
}

static uint64_t run_once(
    const uint8_t *src,
    uint8_t *dst,
    size_t total_bytes,
    size_t record_bytes,
    size_t batch_size,
    size_t iterations,
    volatile uint64_t *sink) {
  uint64_t checksum = 0;
  uint64_t start = now_ns();
  for (size_t iter = 0; iter < iterations; ++iter) {
    memcpy(dst, src, total_bytes);
    for (size_t row = 0; row < batch_size; ++row) {
      size_t offset = row * record_bytes;
      checksum += dst[offset];
      checksum ^= dst[offset + record_bytes - 1];
    }
  }
  uint64_t elapsed = now_ns() - start;
  *sink += checksum;
  return elapsed;
}

int main(int argc, char **argv) {
  size_t repeats = 5;
  uint64_t target_bytes = 134217728ull;
  size_t min_iterations = 128;
  if (argc > 1) {
    target_bytes = strtoull(argv[1], NULL, 10);
  }
  if (argc > 2) {
    repeats = strtoull(argv[2], NULL, 10);
  }
  if (argc > 3) {
    min_iterations = strtoull(argv[3], NULL, 10);
  }
  if (repeats < 3) {
    repeats = 3;
  }
  if (repeats > 16) {
    repeats = 16;
  }

  printf("{\"target_bytes_per_repeat\":%llu,\"repeats\":%zu,\"rows\":[", (unsigned long long)target_bytes, repeats);
  int first = 1;
  volatile uint64_t sink = 0;
  for (size_t profile_index = 0; profile_index < sizeof(PROFILES) / sizeof(PROFILES[0]); ++profile_index) {
    profile_t profile = PROFILES[profile_index];
    for (size_t batch_index = 0; batch_index < sizeof(BATCHES) / sizeof(BATCHES[0]); ++batch_index) {
      size_t batch_size = BATCHES[batch_index];
      size_t total_bytes = profile.record_bytes * batch_size;
      uint8_t *src = NULL;
      uint8_t *dst = NULL;
      if (posix_memalign((void **)&src, 64, total_bytes + 64) != 0) {
        return 2;
      }
      if (posix_memalign((void **)&dst, 64, total_bytes + 64) != 0) {
        free(src);
        return 2;
      }
      fill_source(src, total_bytes, profile.record_bytes);
      memset(dst, 0, total_bytes + 64);
      size_t iterations = iterations_for(total_bytes, target_bytes, min_iterations);
      (void)run_once(src, dst, total_bytes, profile.record_bytes, batch_size, 8, &sink);
      double per_request_ns[16];
      for (size_t repeat = 0; repeat < repeats; ++repeat) {
        uint64_t elapsed = run_once(src, dst, total_bytes, profile.record_bytes, batch_size, iterations, &sink);
        per_request_ns[repeat] = (double)elapsed / (double)(iterations * batch_size);
      }
      double sum = 0.0;
      for (size_t repeat = 0; repeat < repeats; ++repeat) {
        sum += per_request_ns[repeat];
      }
      double mean = sum / (double)repeats;
      double p50 = percentile(per_request_ns, repeats, 0.50);
      double p95 = percentile(per_request_ns, repeats, 0.95);
      double variance = 0.0;
      for (size_t repeat = 0; repeat < repeats; ++repeat) {
        double diff = per_request_ns[repeat] - mean;
        variance += diff * diff;
      }
      variance /= (double)repeats;
      double cv = mean == 0.0 ? 0.0 : (variance > 0.0 ? __builtin_sqrt(variance) / mean : 0.0);
      if (!first) {
        printf(",");
      }
      first = 0;
      printf("{\"profile\":\"%s\",\"record_bytes\":%zu,\"batch_size\":%zu,\"iterations\":%zu,"
             "\"total_bytes_per_iteration\":%zu,\"source_text_exposed\":%s,\"source_kv_exposed\":%s,"
             "\"mean_ns_per_request\":%.6f,\"p50_ns_per_request\":%.6f,\"p95_ns_per_request\":%.6f,"
             "\"cv\":%.8f,\"repeat_ns_per_request\":[",
             profile.name,
             profile.record_bytes,
             batch_size,
             iterations,
             total_bytes,
             profile.source_text_exposed ? "true" : "false",
             profile.source_kv_exposed ? "true" : "false",
             mean,
             p50,
             p95,
             cv);
      for (size_t repeat = 0; repeat < repeats; ++repeat) {
        if (repeat > 0) {
          printf(",");
        }
        printf("%.6f", per_request_ns[repeat]);
      }
      printf("]}");
      free(src);
      free(dst);
    }
  }
  printf("],\"sink\":%llu}\n", (unsigned long long)sink);
  return 0;
}
