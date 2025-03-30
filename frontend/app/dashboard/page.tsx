"use client"
import ProtectedRoute from "@/components/protectedRoute";
import {useEffect, useState} from "react";
import FileInputForm from "@/components/fileinputform";
import axiosInstance from "@/lib/axios";
import {useRouter} from "next/navigation";

interface User {
    email: string;
    id: number;
}

interface Person {
    id: number;
    name: string;
}

const Page = () => {
    const [message, setMessage] = useState<User>({email: "", id: 0});
    const [data, setData] = useState<Person[]>([]);
    const router = useRouter();

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axiosInstance.get("/hello");
                setMessage(response.data);
            } catch (error) {
                console.error("Error fetching data:", error);
                router.push("/login");
            }

            try {
                const data = await axiosInstance.get("people");
                setData(data.data);
            } catch (error) {
                console.error("Error fetching data:", error);
            }
        };

        fetchData();
    }, []);

    return (
        <ProtectedRoute>
            <div className={"flex flex-col items-center justify-center min-w-full min-h-screen"}>
                <p>Email: {message.email}</p>
                <p>ID: {message.id}</p>
                <div className={"w-1/2"}>
                    <FileInputForm></FileInputForm>
                </div>

            </div>
        </ProtectedRoute>
    );
}

export default Page;