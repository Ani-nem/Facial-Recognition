"use client"
import ProtectedRoute from "@/components/protectedRoute";
import {useEffect, useState} from "react";
import axiosInstance from "@/lib/axios";

interface User{
    email: string;
    id: number;
}

interface Person{
    id: number;
    name: string;
}

const Page = () => {
    const [message, setMessage] = useState<User>({email: "", id: 0});
    const [data, setData] = useState([]);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axiosInstance.get("/hello");
                setMessage(response.data);
            } catch (error) {
                console.error("Error fetching data:", error);
            }

            try {
                const data = await axiosInstance.get("people");
                setMessage(data.data);
            } catch (error) {
                console.error("Error fetching data:", error);
            }
        };

        fetchData();
    }, []);

    return (
        <ProtectedRoute>
            <div>
                <p>Email: {message.email}</p>
                <p>ID: {message.id}</p>


            </div>
        </ProtectedRoute>
    );
}

export default Page;