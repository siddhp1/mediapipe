#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float x, y, z;
} MP_Landmark;

typedef struct {
  MP_Landmark landmarks[21];
  int present;   // 1 if detected
  char label[8]; // "Left" or "Right"
  float score;   // confidence [0,1]
} MP_Hand;

typedef struct {
  MP_Hand hands[2]; // up to 2 hands
  int num_hands;
} MP_HandsResult;

// Initialize graph, returns 0 on success
int mp_hands_create(void **ctx);

// Process a frame (BGR 8UC3), returns 0 on success
int mp_hands_process(void *ctx, const unsigned char *bgr, int width, int height,
                     MP_HandsResult *out);

// Cleanup
void mp_hands_destroy(void *ctx);

#ifdef __cplusplus
}
#endif
