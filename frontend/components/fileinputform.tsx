'use client'
import {useState} from "react";
import {Input} from "@/components/ui/input";
import {Card, CardContent, CardDescription, CardHeader, CardTitle} from "@/components/ui/card";
import {Button} from "@/components/ui/button";
import {LucideSend, CircleAlert, Upload, Trash2} from "lucide-react";
import axiosInstance from "@/lib/axios";
import {useToast} from "@/hooks/use-toast";
import {cn} from "@/lib/utils";
import {v4 as uuidv4} from "uuid";

type Image = {
    file: File
    id: string
    preview: string
}

const FileInputForm = () => {
    const [images, setImages] = useState<Image[]>([]);
    const [message, setMessage] = useState("");
    const [isDragging, setIsDragging] = useState(false);
    const {toast} = useToast();

    const handleOnSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (images?.length == 0) {
            return;
        } else {
            console.log('images', images)
            const formData = new FormData();
            images.map((img) => {
                formData.append("files", img.file);
            })

            try {
                const response = await axiosInstance.post("/api/upload_images", formData)
                if (response.status == 200) {
                    setMessage("Images sorted successfully");
                }
            } catch (error) {
                console.log(error);
                setMessage("An error occured");

            }
        }
    }

    const handleClear = (e: React.FormEvent) => {
        e.preventDefault();
        images.map((image) => {
            URL.revokeObjectURL(image.preview)
        })
        setImages([]);
    }
    const handleFileChange = (selectedFiles: FileList | null) => {
        if (selectedFiles == null) {
            return;
        } else {

            const newFiles = Array.from(selectedFiles).filter((file) => {
                return file.type.startsWith("image/");
            }) as File[];

            const newImages: Image[] = newFiles.map((file) => (
                {
                    file,
                    id: uuidv4(),
                    preview: URL.createObjectURL(file)
                }
            ))

            if (newImages.length != selectedFiles.length) {
                toast({
                    description: (
                        <div className={"flex items-center"}>
                            <CircleAlert className="h-8 w-8 mr-2 text-red-700 inline"/>
                            <div>
                                <h1 className={"font-bold text-md"}>Unsupported File Type</h1>
                                <h2 className="font-medium text-sm">
                                    Looks like you uploaded the wrong file!
                                </h2>
                            </div>
                        </div>
                    )
                })
            }


            setImages([...images, ...newImages]);
        }
    }

    const handleFileRemove = (id: string) => {
        const new_images = images.filter((img) => {
            if (img.id == id) {
                URL.revokeObjectURL(img.preview);
                return false;
            }
            return true;
        })

        setImages(new_images);
    }

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(true)
    }

    const handleDragLeave = () => {
        setIsDragging(false)
    }

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(false)
        handleFileChange(e.dataTransfer.files)
    }

    return (
        <Card className={"flex flex-col"}>
            <CardHeader>
                <CardTitle className={"font-semibold text-2xl"}>Upload</CardTitle>
                <CardDescription className={"font-medium"}>Upload a couple photos to be sorted</CardDescription>
            </CardHeader>
            <CardContent>
                <form onSubmit={handleOnSubmit}>
                    <Input
                        className={"hidden"}
                        name="image_input"
                        type={"file"}
                        multiple={true}
                        accept={"image/*"}
                        onChange={(e) => {
                            handleFileChange(e.target.files)
                        }}>
                    </Input>
                    <Card
                        className={cn("p-6 h-[200] border-dashed border-2 cursor-pointer hover:bg-popover", isDragging ? "bg-primary" : "bg-card")}
                        onClick={() => {
                            document.getElementsByName("image_input")[0].click()
                        }}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}>
                        <div className={"flex flex-col items-center justify-around h-full"}>
                            <Upload className={"h-10 w-10 text-secondary-foreground"}></Upload>
                            <h2 className={"font-semibold text-md"}>Drag images here or click to browse</h2>
                            <p className={"font-normal text-sm"}>Supports PNG, JPEG</p>
                        </div>
                    </Card>

                    {/*The individual images themselves*/}
                    <Card
                        className={cn("relative", (images.length > 0) ? "border-none shadow-none overflow-hidden" : "hidden")}>
                        <ul className={"grid lg:grid-cols-3 md:grid-cols-2"}>
                            {images.map((img) => (
                                <li key={img.id}>
                                    <Card className={"group aspect-square overflow-hidden m-1 relative"}>
                                        <Button onClick={() => {
                                            handleFileRemove(img.id)
                                        }} type={"button"} variant={"outline"}
                                                className={"opacity-0 group-hover:opacity-100 hover:bg-popover transition-opacity h-8 w-5 bg-black absolute right-2 top-2"}>
                                            <Trash2 className={"text-red-700"}></Trash2>
                                        </Button>
                                        <img className={"w-full"} src={img.preview}></img>
                                    </Card>
                                </li>
                            ))}
                        </ul>
                    </Card>

                    <div className={cn((images.length > 0) ? "flex" : "hidden")}>
                        <Button className={"w-1/2 bg-red-400 font-semibold rounded-r-none"}
                                type={"reset"}
                                onClick={handleClear}>
                            Clear All</Button>
                        <Button className={"w-1/2 rounded-l-none"} type={"submit"}>
                            Submit<LucideSend></LucideSend>
                        </Button>

                    </div>

                </form>
            </CardContent>

        </Card>
    )
}


export default FileInputForm;