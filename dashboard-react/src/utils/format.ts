export function formatBytes(bytes: number, decimals = 1): string {
  if (!bytes || bytes === 0) return '0B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + sizes[i];
}

export function getTemperatureColor(temp: number): string {
  if (isNaN(temp) || temp === null) return 'rgba(179, 179, 179, 0.8)';

  const coolTemp = 45;
  const midTemp = 57.5;
  const hotTemp = 75;

  const coolColor = { r: 93, g: 173, b: 226 };
  const midColor = { r: 255, g: 215, b: 0 };
  const hotColor = { r: 244, g: 67, b: 54 };

  let r: number, g: number, b: number;

  if (temp <= coolTemp) {
    ({ r, g, b } = coolColor);
  } else if (temp <= midTemp) {
    const ratio = (temp - coolTemp) / (midTemp - coolTemp);
    r = Math.round(coolColor.r * (1 - ratio) + midColor.r * ratio);
    g = Math.round(coolColor.g * (1 - ratio) + midColor.g * ratio);
    b = Math.round(coolColor.b * (1 - ratio) + midColor.b * ratio);
  } else if (temp < hotTemp) {
    const ratio = (temp - midTemp) / (hotTemp - midTemp);
    r = Math.round(midColor.r * (1 - ratio) + hotColor.r * ratio);
    g = Math.round(midColor.g * (1 - ratio) + hotColor.g * ratio);
    b = Math.round(midColor.b * (1 - ratio) + hotColor.b * ratio);
  } else {
    ({ r, g, b } = hotColor);
  }

  return `rgb(${r}, ${g}, ${b})`;
}
