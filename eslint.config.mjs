import nextConfig from "eslint-config-next";

const config = [
  {
    ignores: [".venv/**"],
  },
  ...nextConfig,
];

export default config;
