"use client"
import ProtectedRoute from "@/components/protectedRoute";
import {useEffect, useState} from "react";
import FileInputForm from "@/components/fileinputform";
import GroupedImages from "@/components/groupedImages";
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
            } catch (error) {
                console.error("Error fetching data:", error);
            }
        };

        fetchData();
    }, []);

    return (
        <ProtectedRoute>
            <div className={"flex flex-col items-center min-w-full min-h-screen pt-16"}>
                <div className={"w-1/2"}>
                    <FileInputForm></FileInputForm>
                </div>
            </div>
            <GroupedImages>

            </GroupedImages>
        </ProtectedRoute>


    );
}

export default Page;