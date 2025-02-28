"use client"
import {useRouter} from 'next/navigation'
import {get_access_token, get_user_info, logout, UserContext} from "@/utils/auth";
import React, {useContext, useEffect, useState} from "react";

interface ProtectedRouteProps{
    children: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({children}) => {
    const router = useRouter();
    const context = useContext(UserContext);
    const [loading, setLoading] = useState(true);


    useEffect(() => {
        const checkAuth = async () => {
            const token = get_access_token();
            if (!token) {
                router.push("/login");
                return;
            }

            if (!context?.user) {
                try {
                    const userInfo = await get_user_info();
                    context?.setUser(userInfo);
                } catch (error) {
                    logout();
                    router.push("/login");
                }
            }

            setLoading(false);
        };
        checkAuth();
    }, [context, router]);

    if (loading) {
        return <div>Loading...</div>;
    }
    if (context?.user) {
        return <>{children}</>;
    }

    return <div>An error occurred</div>

}

export default ProtectedRoute;