"use client"
import React, {ReactNode, useEffect, useState} from "react";
import {get_user_info, User, UserContext} from "@/utils/auth";

interface AuthProviderProps{
    children: ReactNode;
}
export const AuthProvider: React.FC<AuthProviderProps>  = ({children}) => {
    const [user, setUser] = useState<User | null>(null);
    useEffect(() => {
        const fetchUser = async () => {
            const userData = await get_user_info();
            setUser(userData);
        };
        fetchUser();
    }, []);

    return (
        <UserContext.Provider value={{user, setUser}}>
            {children}
        </UserContext.Provider>
    )

}

export default AuthProvider;
