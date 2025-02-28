import {createContext} from "react";
import axiosInstance from "@/lib/axios";


export interface User{
    id: number;
    email: string;
}

interface UserContextProps{
    user: User | null;
    setUser: (user: User | null) => void;
}

export const UserContext = createContext<UserContextProps | null>(null);

export const login = async(email: string, password :string): Promise<Boolean> => {
    try {
       const formData = new FormData();
       formData.append("username", email);
       formData.append("password", password);
       const response = await axiosInstance.post("auth/token", formData, {
           headers: {'Content-Type' : 'application/x-www-form-urlencoded'},
       });

       if (response.status === 200) {
            const { access_token } = response.data;
            localStorage.setItem('access_token', access_token);
            return true;
       } else {
           console.log("something happened!")
           return false;
       }
    } catch (exception){
        console.log(exception);
        return false;
    }
};

export const logout = () => {
    localStorage.removeItem('access_token');
};

export const get_access_token = (): string | null => {
    return localStorage.getItem("access_token");
};

export const isAuthenticated = () => {
    !!get_access_token();
}

export const get_user_info = async ():Promise<User | null> =>{
    const token = get_access_token();
    if (token){
        try {
            const response = await axiosInstance.get("auth/me", {
                headers: {'Authorization' : `Bearer ${token}`},
            });

            if (response.status === 200){
                return response.data;
            } else {
                console.log("Error occurred while fetching user data: ", response.data);
                return null;
            }
        } catch (exception){
            console.log("Error occurred while fetching user data: ", exception);
            return null;
        }
    }
    return null;
};
