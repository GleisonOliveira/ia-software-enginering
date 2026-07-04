module.exports = {
  testEnvironment: 'node',
  preset: 'ts-jest/presets/default-esm',
  moduleFileExtensions: ['ts', 'js', 'json'],
  testMatch: ['**/tests/**/*.test.ts'],
  transformIgnorePatterns: ['node_modules'],
};
