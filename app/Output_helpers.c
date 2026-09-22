/**
 * @file output_helpers.c
 * @brief Implementation of general-purpose helper functions used in Output subsystem.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include "Output_helpers.h"
#include "Model.h"
#include "cJSON.h"

// --- base64 encoder table ---
static const char base64_table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/**
 * @brief Encode a memory buffer to base64. Allocates a new string.
 */
char* base64_encode(const unsigned char *src, size_t len)
{
    if (!src || len == 0) return NULL;

    size_t olen = 4 * ((len + 2) / 3);     // Output is always a multiple of 4
    char *out = (char*)malloc(olen + 1);
    if (!out) return NULL;
    char *pos = out;

    int val = 0, valb = -6;
    for (size_t i = 0; i < len; ++i) {
        val = (val << 8) + src[i];
        valb += 8;
        while (valb >= 0) {
            *pos++ = base64_table[(val >> valb) & 0x3F];
            valb -= 6;
        }
    }
    if (valb > -6) *pos++ = base64_table[((val << 8) >> (valb + 8)) & 0x3F];
    while ((pos - out) % 4) *pos++ = '=';
    *pos = '\0';

    return out;
}

/**
 * @brief Replace all spaces in a null-terminated string with underscores (modifies in-place).
 */
void replace_spaces(char *str)
{
    if (!str) return;
    while (*str) {
        if (*str == ' ')
            *str = '_';
        ++str;
    }
}

