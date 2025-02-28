import axios from 'axios';
import { get_access_token } from '@/utils/auth';


const axiosInstance = axios.create({
    baseURL: process.env.NEXT_PUBLIC_API_URL
});


axiosInstance.interceptors.request.use(
    (config) => {
        const token = get_access_token();
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => Promise.reject(error)
);

export default axiosInstance;