/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stm32f4xx_hal_flash.h"
#include <stdint.h>
#include <string.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

// PA2 high
static void trigger_1()
{
	HAL_GPIO_Write_PIN(GPIOA, GPIO_PIN_2, GPIO_PIN_SET);
}

// PA2 low
static void trigger_0()
{
	HAL_GPIO_Write_PIN(GPIOA, GPIO_PIN_2, GPIO_PIN_RESET);
}

static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

// --- Definitions ---
#define AES_BLOCK_SIZE 16
#define AES_KEY_SIZE 16 // 128 bits
#define AES_ROUNDS 10

// The State is a 4x4 matrix of bytes, but we use a linear buffer for simplicity.
// state[i] corresponds to the standard's row r and column c: i = r + 4c

// --- Lookup Tables (S-Box, Inverse S-Box, Rcon) ---

static const uint8_t sbox[256] = {
    // 0     1     2     3     4     5     6     7     8     9     A     B     C     D     E     F
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};

static const uint8_t rsbox[256] = {
    // Inverse S-Box
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d};

static const uint8_t Rcon[11] = {
    // Round constants
    0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};

// --- Helper Functions ---

// Get S-box value
#define SBOX(x) (sbox[(x)])
// Get Inverse S-box value
#define RSBOX(x) (rsbox[(x)])

// Galois Field (GF(2^8)) Multiplication x * 2
// Multiplies polynomial x by x in GF(2^8) modulo x^8 + x^4 + x^3 + x + 1 (0x11b)
static uint8_t xtime(uint8_t x)
{
    return (x << 1) ^ ((x >> 7) * 0x1b);
}

// General Galois Field Multiplication
static uint8_t gmul(uint8_t a, uint8_t b)
{
    uint8_t p = 0;
    for (int i = 0; i < 8; i++)
    {
        if (b & 1)
            p ^= a;
        a = xtime(a);
        b >>= 1;
    }
    return p;
}

// --- Key Expansion ---

// Rotates a 4-byte word: [a0, a1, a2, a3] -> [a1, a2, a3, a0]
static void RotWord(uint8_t *word)
{
    uint8_t temp = word[0];
    word[0] = word[1];
    word[1] = word[2];
    word[2] = word[3];
    word[3] = temp;
}

// Replaces each byte in a 4-byte word using the S-box
static void SubWord(uint8_t *word)
{
    for (int i = 0; i < 4; i++)
    {
        word[i] = SBOX(word[i]);
    }
}

// Expands the 128-bit key into the full key schedule (176 bytes)
void KeyExpansion(const uint8_t *key, uint8_t *roundKeys)
{
    // First 16 bytes are the original key
    for (int i = 0; i < AES_KEY_SIZE; i++)
    {
        roundKeys[i] = key[i];
    }

    uint8_t temp[4];
    int i = AES_KEY_SIZE;

    while (i < (AES_BLOCK_SIZE * (AES_ROUNDS + 1)))
    {
        // Copy previous word to temp
        for (int j = 0; j < 4; j++)
        {
            temp[j] = roundKeys[i - 4 + j];
        }

        if (i % AES_KEY_SIZE == 0)
        {
            RotWord(temp);
            SubWord(temp);
            temp[0] ^= Rcon[i / AES_KEY_SIZE];
        }

        for (int j = 0; j < 4; j++)
        {
            roundKeys[i] = roundKeys[i - AES_KEY_SIZE] ^ temp[j];
            i++;
        }
    }
}

// --- Core AES Transformations ---

// AddRoundKey: XOR state with Round Key
static void AddRoundKey(uint8_t *state, const uint8_t *roundKey)
{
    for (int i = 0; i < AES_BLOCK_SIZE; i++)
    {
        state[i] ^= roundKey[i];
    }
}

// SubBytes: Substitute bytes using S-box
static void SubBytes(uint8_t *state)
{
    for (int i = 0; i < AES_BLOCK_SIZE; i++)
    {
        state[i] = SBOX(state[i]);
    }
}

// InvSubBytes: Substitute bytes using Inverse S-box
static void InvSubBytes(uint8_t *state)
{
    for (int i = 0; i < AES_BLOCK_SIZE; i++)
    {
        state[i] = RSBOX(state[i]);
    }
}

// ShiftRows: Shift rows of the state cyclically
// Row 0: No shift
// Row 1: Left shift 1
// Row 2: Left shift 2
// Row 3: Left shift 3
// Note: State is linear, but logically 4x4 column-major.
// Indices:
// 0  4  8 12
// 1  5  9 13
// 2  6 10 14
// 3  7 11 15
static void ShiftRows(uint8_t *state)
{
    uint8_t temp;

    // Row 1: Rotate left by 1
    temp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = temp;

    // Row 2: Rotate left by 2
    temp = state[2];
    state[2] = state[10];
    state[10] = temp;
    temp = state[6];
    state[6] = state[14];
    state[14] = temp;

    // Row 3: Rotate left by 3 (or right by 1)
    temp = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = state[3];
    state[3] = temp;
}

// InvShiftRows: Inverse of ShiftRows
static void InvShiftRows(uint8_t *state)
{
    uint8_t temp;

    // Row 1: Rotate right by 1
    temp = state[13];
    state[13] = state[9];
    state[9] = state[5];
    state[5] = state[1];
    state[1] = temp;

    // Row 2: Rotate right by 2
    temp = state[2];
    state[2] = state[10];
    state[10] = temp;
    temp = state[6];
    state[6] = state[14];
    state[14] = temp;

    // Row 3: Rotate right by 3 (or left by 1)
    temp = state[3];
    state[3] = state[7];
    state[7] = state[11];
    state[11] = state[15];
    state[15] = temp;
}

// MixColumns: Matrix multiplication
// [02 03 01 01]
// [01 02 03 01]
// [01 01 02 03]
// [03 01 01 02]
static void MixColumns(uint8_t *state)
{
    uint8_t tmp[4];
    // Process each column (0, 4, 8, 12)
    for (int i = 0; i < 4; i++)
    {
        int idx = i * 4;
        uint8_t s0 = state[idx + 0];
        uint8_t s1 = state[idx + 1];
        uint8_t s2 = state[idx + 2];
        uint8_t s3 = state[idx + 3];

        state[idx + 0] = xtime(s0) ^ (xtime(s1) ^ s1) ^ s2 ^ s3;         // 2*s0 ^ 3*s1 ^ s2 ^ s3
        state[idx + 1] = s0 ^ xtime(s1) ^ (xtime(s2) ^ s2) ^ s3;         // s0 ^ 2*s1 ^ 3*s2 ^ s3
        state[idx + 2] = s0 ^ s1 ^ xtime(s2) ^ (xtime(s3) ^ s3);         // s0 ^ s1 ^ 2*s2 ^ 3*s3
        state[idx + 3] = (xtime(s0) ^ s0) ^ s1 ^ s2 ^ xtime(s3);         // 3*s0 ^ s1 ^ s2 ^ 2*s3
    }
}

// InvMixColumns: Inverse Matrix multiplication
// [0e 0b 0d 09]
// [09 0e 0b 0d]
// [0d 09 0e 0b]
// [0b 0d 09 0e]
static void InvMixColumns(uint8_t *state)
{
    for (int i = 0; i < 4; i++)
    {
        int idx = i * 4;
        uint8_t s0 = state[idx + 0];
        uint8_t s1 = state[idx + 1];
        uint8_t s2 = state[idx + 2];
        uint8_t s3 = state[idx + 3];

        state[idx + 0] = gmul(0x0e, s0) ^ gmul(0x0b, s1) ^ gmul(0x0d, s2) ^ gmul(0x09, s3);
        state[idx + 1] = gmul(0x09, s0) ^ gmul(0x0e, s1) ^ gmul(0x0b, s2) ^ gmul(0x0d, s3);
        state[idx + 2] = gmul(0x0d, s0) ^ gmul(0x09, s1) ^ gmul(0x0e, s2) ^ gmul(0x0b, s3);
        state[idx + 3] = gmul(0x0b, s0) ^ gmul(0x0d, s1) ^ gmul(0x09, s2) ^ gmul(0x0e, s3);
    }
}

// --- Main Cipher Functions ---

// Encrypt a single 128-bit block
void AES_Encrypt(const uint8_t *roundKeys, const uint8_t *input, uint8_t *output)
{
    uint8_t state[AES_BLOCK_SIZE];

    // Copy input to state
    memcpy(state, input, AES_BLOCK_SIZE);

    // Initial Round
    AddRoundKey(state, roundKeys); // Round Key 0

    // 9 Main Rounds
    for (int round = 1; round < AES_ROUNDS; round++)
    {
        SubBytes(state);
        ShiftRows(state);
        MixColumns(state);
        AddRoundKey(state, roundKeys + (round * AES_BLOCK_SIZE));
    }

    // Final Round (No MixColumns)
    SubBytes(state);
    ShiftRows(state);
    AddRoundKey(state, roundKeys + (AES_ROUNDS * AES_BLOCK_SIZE));

    // Copy state to output
    memcpy(output, state, AES_BLOCK_SIZE);
}

// Decrypt a single 128-bit block
void AES_Decrypt(const uint8_t *roundKeys, const uint8_t *input, uint8_t *output)
{
    uint8_t state[AES_BLOCK_SIZE];

    memcpy(state, input, AES_BLOCK_SIZE);

    // Initial Round (Use last round key)
    AddRoundKey(state, roundKeys + (AES_ROUNDS * AES_BLOCK_SIZE));

    // 9 Main Rounds (Reverse order)
    for (int round = AES_ROUNDS - 1; round > 0; round--)
    {
        InvShiftRows(state);
        InvSubBytes(state);
        AddRoundKey(state, roundKeys + (round * AES_BLOCK_SIZE));
        InvMixColumns(state);
    }

    // Final Round
    InvShiftRows(state);
    InvSubBytes(state);
    AddRoundKey(state, roundKeys); // Round Key 0

    memcpy(output, state, AES_BLOCK_SIZE);
}





/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART1_UART_Init();
  /* USER CODE BEGIN 2 */

  dwt_init();
  uint32_t t0 = 0;
  uint32_t t1 = 0;
  uint32_t cycles = 0;

  uint8_t key[16] = {
  		0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
  		0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};

  uint8_t input[16] = {
	0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
	0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34};

  uint8_t encrypted[16];


  uint8_t roundKeys[176];

  KeyExpansion(key, roundKeys);

  	// warm-up

  for(uint8_t i = 0; i < 200; i++)
  {
	AES_Encrypt(roundKeys, input, encrypted);
	memcpy(input, encrypted, AES_BLOCK_SIZE);
  }
  uint32_t counter = 0;

  uint32_t tx_data[2];

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */


	  t0 = DWT->CYCCNT;
	  AES_Encrypt(roundKeys, input, encrypted);
	  t1 = DWT->CYCCNT;
	  memcpy(input, encrypted, AES_BLOCK_SIZE);
	  if(t1 > t0) cycles = t1 - t0;
      else cycles = (UINT32_MAX - t0) + t1;
	  counter++;
	  tx_data[0] = counter;
	  tx_data[1] = cycles;
	  HAL_UART_Transmit(&huart1, (uint8_t*)tx_data, 8, 100);

	  HAL_Delay(100);
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
