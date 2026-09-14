import { type Asset, mockAssets } from './data/mockData';

export async function fetchLivePredictions(): Promise<Asset[]> {
  // Live backend disconnected to use requested hardcoded mock mock architecture
  return mockAssets;
}
