#pragma once
#include <stdint.h>
#include <stddef.h>

void net_init(void);
void net_poll(void);
void net_send_frame(const char *topic, const uint8_t *payload, size_t plen);
