import { apiSlice } from '../api';

/** Existing credential state; active does not imply online presence. */
export interface OperatorDevice {
  deviceId: string;
  name: string;
  pairedAt: string;
  refreshExpiresAt: string | null;
  state: 'active' | 'revoked';
  current: boolean;
}

const headers = { 'X-Skulk-Dashboard': 'pairing-v1' };
const devicesApi = apiSlice.injectEndpoints({
  endpoints: build => ({
    getOperatorDevices: build.query<{ devices: OperatorDevice[] }, void>({
      query: () => ({ url: '/v1/auth/devices', headers }),
      providesTags: ['OperatorDevices'],
    }),
    revokeOperatorDevice: build.mutation<void, string>({
      query: deviceId => ({ url: `/v1/auth/devices/${encodeURIComponent(deviceId)}`, method: 'DELETE', headers }),
      invalidatesTags: ['OperatorDevices'],
    }),
  }),
});

export const { useGetOperatorDevicesQuery, useRevokeOperatorDeviceMutation } = devicesApi;
