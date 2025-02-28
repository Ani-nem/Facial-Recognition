"use client";
import {useState, useContext} from "react";
import {Button} from "@/components/ui/button";
import {Input} from "@/components/ui/input";
import {Card, CardContent, CardHeader, CardTitle} from "@/components/ui/card";
import {useRouter} from "next/navigation";
import {get_user_info, login, UserContext} from "@/utils/auth";
import {Label} from "@/components/ui/label";

const LoginForm = () => {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const router = useRouter();
    const context = useContext(UserContext);

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!email || !password) {
            setError("Missing login or password")
        } else {
            setLoading(true);
            setError("");
            const success = await login(email, password);
            setLoading(false);

            if (success) {
                const userInfo = await get_user_info();
                context?.setUser(userInfo);
                router.push("/dashboard");
            } else {
                setError("Incorrect Email or Password");
            }
        }

    };

    return (
        <Card className="w-full max-w-sm">
            <CardHeader className={"flex flex-row justify-center"}>
                <CardTitle className={"text-2xl"}>Login</CardTitle>
            </CardHeader>
            <CardContent>
                <form onSubmit={handleLogin} noValidate={true}>
                    <div className="space-y-4">
                        <div className={"grid gap-2"}>
                            <Label>Email</Label>
                            <Input
                                type="email"
                                placeholder="Email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                            />
                        </div>

                        <div className={"grid gap-2"}>
                            <Label>Password</Label>
                            <Input
                                type="password"
                                placeholder="Password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                            />
                        </div>

                        {error && (
                            <div className="text-red-700 text-sm">{error}</div>
                        )}
                        <Button type="submit" className="w-full" disabled={loading}>
                            {loading ? "Logging in..." : "Login"}
                        </Button>
                    </div>
                </form>
            </CardContent>
        </Card>
    );
}

export default LoginForm;